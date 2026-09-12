"""CogentDataset for loading and preprocessing spatial transcriptomics data."""

import contextlib
import gc
import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.utils.data
from numba import njit, prange
from sklearn.neighbors import BallTree
from tqdm import tqdm

from cogent.utils import normalize_expression


@njit(parallel=True)
def _build_adjacency_components(neighbor_indptr: np.ndarray):
    """Build row indices and weights for adjacency matrix."""
    n_cells = len(neighbor_indptr) - 1
    n_edges = neighbor_indptr[-1]

    row_indices = np.empty(n_edges, dtype=np.int64)
    expanded_weights = np.empty(n_edges, dtype=np.float32)

    for i in prange(n_cells):
        start = neighbor_indptr[i]
        end = neighbor_indptr[i + 1]
        degree = end - start
        weight = 1.0 / max(degree, 1)
        for j in range(start, end):
            row_indices[j] = i
            expanded_weights[j] = weight

    return row_indices, expanded_weights


class CogentDataset:
    """Dataset for Cogent training with flexible storage modes.

    Supports both in-memory and HDF5-backed storage for efficient training
    on spatial transcriptomics data. Implements PyTorch Dataset interface.

    Args:
        adatas: List of AnnData objects or h5ad file paths, or None
        genes: List of gene names to include
        groups: Optional subset of groups to include
        path: Path to existing HDF5 dataset (for loading)
        in_memory: If True, store data in memory; if False, use HDF5
        spatial_conv: Radii for spatial convolution (None for no spatial)
        normalize: Whether to normalize expression data
        group_name: Column name for group labels in adata.obs
        sample_name: Column name for sample labels in adata.obs
        cell_batch_size: Batch size for training
        chunk_size: HDF5 chunk size
    """

    def __init__(
        self,
        adatas: Sequence[sc.AnnData] | Sequence[str] | None = None,
        genes: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        path: str | Path | None = None,
        in_memory: bool = False,
        group_name: str = "group",
        sample_name: str = "sample",
        cell_batch_size: int = 512,
        chunk_size: int = 8192,
    ):
        if path is not None:
            self._load_from_path(Path(path), in_memory)
        elif adatas is not None:
            if genes is None:
                raise ValueError("genes must be provided when adatas is given")
            self._adatas = adatas
            self._genes = genes
            self._groups = groups
            self._in_memory = in_memory
            self._group_name = group_name
            self._sample_name = sample_name
            self._cell_batch_size = cell_batch_size
            self._chunk_size = chunk_size
            self._preprocessed = False

            self._gene_names = list(genes)
            self._n_genes = len(genes)

            self._data = None
            self._h5file = None
            self._path = None
        else:
            raise ValueError("Either path or adatas must be provided")

    def preprocess(
        self,
        spatial_conv: Sequence[float] | None = None,
        normalize: bool = False,
    ):
        """Preprocess data with spatial convolution and normalization.

        Args:
            spatial_conv: Radii for spatial convolution (None for no spatial features)
            normalize: Whether to normalize expression data

        Raises:
            RuntimeError: If dataset was loaded from path or already preprocessed
        """
        if hasattr(self, "_preprocessed") and self._preprocessed:
            raise RuntimeError("Dataset already preprocessed")
        if not hasattr(self, "_adatas"):
            raise RuntimeError("Cannot preprocess dataset loaded from path")

        self._preprocess_from_adatas(
            adatas=self._adatas,
            genes=self._genes,
            groups=self._groups,
            in_memory=self._in_memory,
            spatial_conv=spatial_conv,
            normalize=normalize,
            group_name=self._group_name,
            sample_name=self._sample_name,
            cell_batch_size=self._cell_batch_size,
            chunk_size=self._chunk_size,
        )
        self._preprocessed = True

        del self._adatas
        del self._genes
        del self._groups
        del self._group_name
        del self._sample_name

    def _load_from_path(self, path: Path, in_memory: bool):
        """Load dataset from existing HDF5 file."""
        self._path = path
        self._mode = "memory" if in_memory else "hdf5"

        metadata_path = path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self._n_genes = metadata["n_genes"]
        self._n_groups = metadata["n_groups"]
        self._n_channels = metadata["n_channels"]
        self._gene_names = metadata["genes"]
        self._group_ids = metadata["group_names"]
        self._cell_batch_size = metadata["cell_batch_size"]
        self._chunk_size = metadata.get("chunk_size", 8192)
        self._spatial_conv = metadata.get("spatial_conv")

        if in_memory:
            self._load_all_to_memory()
            self._h5file = None
        else:
            self._data = None
            h5_path = path / "data.h5"
            self._h5file = h5py.File(
                h5_path,
                "r",
                locking=False,
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_w0=0.75,
                rdcc_nslots=10007,
            )

    def _load_all_to_memory(self):
        """Load entire dataset from HDF5 to memory."""
        h5_path = self._path / "data.h5"
        with h5py.File(h5_path, "r") as f:
            boundaries = f["metadata/sample_boundaries"][:]
            cell_to_group = f["metadata/cell_to_group"][:]

            data_list = []
            for i in range(len(boundaries) - 1):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                group_id = int(cell_to_group[start])

                channels_data = self._load_channels_from_h5(f, start, end)
                data_list.append((group_id, channels_data))

        self._data = data_list

    def _load_channels_from_h5(self, f: h5py.File, start: int, end: int) -> np.ndarray:
        """Load all channels for a cell range from HDF5."""
        n_cells = end - start
        result = np.zeros((n_cells, self._n_genes, self._n_channels), dtype=np.float32)

        indptr = f["expression/indptr"][start : end + 1]
        data_start = indptr[0]
        data_end = indptr[-1]
        data = f["expression/data"][data_start:data_end]
        indices = f["expression/indices"][data_start:data_end]
        indptr_adj = indptr - indptr[0]

        expr = sp.csr_matrix(
            (data, indices, indptr_adj),
            shape=(n_cells, self._n_genes),
            dtype=np.float32,
        )
        result[:, :, 0] = expr.toarray()

        for c in range(1, self._n_channels):
            grp = f[f"convolved/channel_{c}"]
            indptr = grp["indptr"][start : end + 1]
            data_start = indptr[0]
            data_end = indptr[-1]
            data = grp["data"][data_start:data_end]
            indices = grp["indices"][data_start:data_end]
            indptr_adj = indptr - indptr[0]

            conv = sp.csr_matrix(
                (data, indices, indptr_adj),
                shape=(n_cells, self._n_genes),
                dtype=np.float32,
            )
            result[:, :, c] = conv.toarray()

        return result

    def _preprocess_from_adatas(
        self,
        adatas: Sequence[sc.AnnData] | Sequence[str],
        genes: Sequence[str],
        groups: Sequence[str] | None,
        in_memory: bool,
        spatial_conv: Sequence[float] | None,
        normalize: bool,
        group_name: str,
        sample_name: str,
        cell_batch_size: int,
        chunk_size: int,
    ):
        """Preprocess AnnData objects into dataset."""
        self._mode = "memory" if in_memory else "hdf5"
        self._gene_names = list(genes)
        self._n_genes = len(genes)
        self._cell_batch_size = cell_batch_size

        if spatial_conv is None:
            self._n_channels = 1
            self._spatial_conv = []
        else:
            self._spatial_conv = list(spatial_conv)
            self._n_channels = 1 + len(self._spatial_conv)

        var_id_map = dict(zip(genes, np.arange(self._n_genes), strict=False))
        group_id_mapping = {}
        sample_id_mapping = {}
        processed_samples = []

        if in_memory:
            self._data = []
            self._h5file = None
            self._process_adatas_to_memory(
                adatas,
                genes,
                groups,
                var_id_map,
                group_id_mapping,
                sample_id_mapping,
                processed_samples,
                normalize,
                group_name,
                sample_name,
            )
        else:
            import tempfile

            self._path = Path(tempfile.mkdtemp(prefix="cogent_"))
            self._data = None
            self._process_adatas_to_hdf5(
                adatas,
                genes,
                groups,
                var_id_map,
                group_id_mapping,
                sample_id_mapping,
                processed_samples,
                normalize,
                group_name,
                sample_name,
                chunk_size,
            )

        self._n_groups = len(group_id_mapping)
        self._group_ids = list(group_id_mapping.keys())
        self._chunk_size = chunk_size

        if not in_memory:
            metadata = {
                "n_channels": self._n_channels,
                "cell_batch_size": cell_batch_size,
                "chunk_size": chunk_size,
                "n_genes": self._n_genes,
                "n_groups": self._n_groups,
                "n_samples": len(sample_id_mapping),
                "group_names": list(group_id_mapping.keys()),
                "sample_names": processed_samples,
                "genes": self._gene_names,
                "spatial_conv": self._spatial_conv,
                "normalized": normalize,
            }

            with open(self._path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            h5_path = self._path / "data.h5"
            self._h5file = h5py.File(
                h5_path,
                "r",
                locking=False,
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_w0=0.75,
                rdcc_nslots=10007,
            )

    def _iter_samples(
        self,
        adatas,
        genes,
        groups,
        var_id_map,
        group_id_mapping,
        sample_id_mapping,
        processed_samples,
        group_name,
        sample_name,
    ):
        """Iterate over all samples from adatas with group/sample tracking.

        Yields:
            (adata, group_id, sample_id, cell_indices, is_new_adata)
        """
        for _, adata in enumerate(adatas):
            if isinstance(adata, (str, Path)):
                adata = sc.read_h5ad(adata)

            unique_groups = np.unique(adata.obs[group_name])
            is_first_sample = True

            for group_label in unique_groups:
                if groups is not None and group_label not in groups:
                    continue

                if group_label not in group_id_mapping:
                    group_id_mapping[group_label] = len(group_id_mapping)
                group_id = group_id_mapping[group_label]

                adata_group = adata[adata.obs[group_name] == group_label]
                unique_samples = np.unique(adata_group.obs[sample_name])

                for sample_label in unique_samples:
                    cell_mask = (adata.obs[group_name] == group_label) & (
                        adata.obs[sample_name] == sample_label
                    )
                    cell_indices = np.where(cell_mask.to_numpy())[0]

                    if len(cell_indices) == 0:
                        continue

                    if sample_label not in sample_id_mapping:
                        sample_id_mapping[sample_label] = len(sample_id_mapping)
                        processed_samples.append(sample_label)
                    sample_id = sample_id_mapping[sample_label]

                    yield adata, group_id, sample_id, cell_indices, is_first_sample
                    is_first_sample = False

    def _process_sample(self, adata, cell_indices, genes, var_id_map, normalize):
        """Process a single sample: extract expression and compute convolutions."""
        X = self._extract_expression(adata, cell_indices, genes, var_id_map, normalize)

        convolutions = None
        spatial_coords = None
        if "spatial" in adata.obsm and self._spatial_conv:
            spatial_coords = adata.obsm["spatial"][cell_indices]
            if not np.any(np.isnan(spatial_coords)):
                convolutions = self._compute_convolutions(X, spatial_coords)
            else:
                spatial_coords = None

        return X, convolutions, spatial_coords

    def _process_adatas_to_memory(
        self,
        adatas,
        genes,
        groups,
        var_id_map,
        group_id_mapping,
        sample_id_mapping,
        processed_samples,
        normalize,
        group_name,
        sample_name,
    ):
        """Process AnnData objects to in-memory storage."""
        pbar = tqdm(total=len(adatas), desc="Loading to memory")

        for (
            adata,
            group_id,
            _sample_id,
            cell_indices,
            is_new_adata,
        ) in self._iter_samples(
            adatas,
            genes,
            groups,
            var_id_map,
            group_id_mapping,
            sample_id_mapping,
            processed_samples,
            group_name,
            sample_name,
        ):
            if is_new_adata:
                pbar.update(1)

            X, convolutions, _ = self._process_sample(
                adata, cell_indices, genes, var_id_map, normalize
            )

            channels_data = self._combine_channels(X, convolutions)
            self._data.append((group_id, channels_data))

            gc.collect()

        pbar.close()

    def _process_adatas_to_hdf5(
        self,
        adatas,
        genes,
        groups,
        var_id_map,
        group_id_mapping,
        sample_id_mapping,
        processed_samples,
        normalize,
        group_name,
        sample_name,
        chunk_size,
    ):
        """Process AnnData objects to HDF5 storage."""
        from cogent.dataset import CogentStorageWriter

        pbar = tqdm(total=len(adatas), desc="Preprocessing to HDF5")

        with CogentStorageWriter(
            self._path,
            n_genes=self._n_genes,
            n_channels=self._n_channels,
            n_groups=0,
            spatial_conv=self._spatial_conv,
            chunk_size=chunk_size,
        ) as writer:
            for (
                adata,
                group_id,
                sample_id,
                cell_indices,
                is_new_adata,
            ) in self._iter_samples(
                adatas,
                genes,
                groups,
                var_id_map,
                group_id_mapping,
                sample_id_mapping,
                processed_samples,
                group_name,
                sample_name,
            ):
                if is_new_adata:
                    pbar.update(1)

                X, convolutions, spatial_coords = self._process_sample(
                    adata, cell_indices, genes, var_id_map, normalize
                )

                writer.add_sample(
                    expression=X,
                    group_id=group_id,
                    sample_id=sample_id,
                    convolved=convolutions,
                    spatial_coords=spatial_coords,
                )

                gc.collect()

        pbar.close()

        with h5py.File(self._path / "data.h5", "r+") as f:
            f.attrs["n_groups"] = len(group_id_mapping)

    def _extract_expression(
        self, adata, cell_indices, genes, var_id_map, normalize
    ) -> sp.csr_matrix:
        """Extract and remap expression matrix."""
        cur_genes = adata.var_names.intersection(genes)
        cur_gene_ids = [var_id_map[g] for g in cur_genes]

        X = adata[cell_indices][:, cur_genes].X
        X = sp.coo_matrix(X) if not sp.issparse(X) else X.tocoo()

        gene_id_mapping = np.zeros(len(cur_gene_ids), dtype=np.int32)
        gene_id_mapping[np.arange(len(cur_gene_ids))] = cur_gene_ids

        X_remapped = sp.coo_matrix(
            (X.data, (X.row, gene_id_mapping[X.col])),
            shape=(len(cell_indices), self._n_genes),
            dtype=np.float32,
        ).tocsr()

        return normalize_expression(X_remapped, normalize)

    def _compute_convolutions(
        self, expression: sp.csr_matrix, spatial_coords: np.ndarray
    ) -> list[sp.csr_matrix]:
        """Compute spatial convolutions for all radii."""
        cache = SpatialNeighborCache(self._spatial_conv)
        neighbor_data = cache.compute_all_radii(0, spatial_coords)

        convolutions = []
        for indices, indptr in neighbor_data:
            conv = compute_spatial_convolution(expression, indices, indptr)
            convolutions.append(conv)

        return convolutions

    def _combine_channels(
        self, expression: sp.csr_matrix, convolutions: list[sp.csr_matrix] | None
    ) -> np.ndarray:
        """Combine expression and convolutions into multi-channel array."""
        n_cells = expression.shape[0]
        result = np.zeros((n_cells, self._n_genes, self._n_channels), dtype=np.float32)
        result[:, :, 0] = expression.toarray()

        if convolutions:
            for c, conv in enumerate(convolutions, start=1):
                result[:, :, c] = conv.toarray()

        return result

    def __len__(self) -> int:
        """Return number of samples."""
        if self._mode == "memory":
            return len(self._data)
        else:
            boundaries = self._h5file["metadata/sample_boundaries"][:]
            return len(boundaries) - 1

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over batches of (group_id, expression_tensor).

        Implements chunked shuffling for HDF5 mode and simple shuffling
        for memory mode.

        Yields:
            Tuple of (group_id, expression) where expression is
            shape (cell_batch_size, n_genes, n_channels)

        Raises:
            ValueError: If cell_batch_size is larger than the smallest group
        """
        min_cells = self._get_min_cells_per_group()
        if self._cell_batch_size > min_cells:
            raise ValueError(
                f"cell_batch_size ({self._cell_batch_size}) is larger than the smallest "
                f"group's cell count ({min_cells}). Use a smaller cell_batch_size."
            )

        if self._mode == "memory":
            yield from self._iterate_memory()
        else:
            yield from self._iterate_hdf5()

    def _get_min_cells_per_group(self) -> int:
        """Get the minimum number of cells across all groups."""
        if self._mode == "memory":
            return min(data.shape[0] for _, data in self._data)
        else:
            boundaries = self._h5file["metadata/sample_boundaries"][:]
            cell_to_group = self._h5file["metadata/cell_to_group"][:]

            group_cells = defaultdict(int)
            for i in range(len(boundaries) - 1):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                group_id = int(cell_to_group[start])
                group_cells[group_id] += end - start

            return min(group_cells.values())

    def _iterate_memory(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over in-memory data."""
        indices = list(range(len(self._data)))
        random.shuffle(indices)

        for idx in indices:
            group_id, data = self._data[idx]
            n_cells = data.shape[0]

            perm = np.random.permutation(n_cells)
            data = data[perm]

            for i in range(0, n_cells, self._cell_batch_size):
                batch = data[i : i + self._cell_batch_size]
                if batch.shape[0] < self._cell_batch_size:
                    needed = self._cell_batch_size - batch.shape[0]
                    wraparound = data[:needed]
                    batch = np.concatenate([batch, wraparound], axis=0)
                has_spatial = self._n_channels > 1 and bool(batch[:, :, 1:].any())
                yield group_id, batch, has_spatial

    def _iterate_hdf5(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over HDF5 data with chunked shuffling."""
        if not hasattr(self, "_cached_group_ranges"):
            boundaries = self._h5file["metadata/sample_boundaries"][:]
            cell_to_group = self._h5file["metadata/cell_to_group"][:]

            group_ranges = defaultdict(list)
            for i in range(len(boundaries) - 1):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                group_id = int(cell_to_group[start])
                group_ranges[group_id].append((start, end))

            self._cached_group_ranges = group_ranges

        group_ids = list(self._cached_group_ranges.keys())
        random.shuffle(group_ids)

        for group_id in group_ids:
            ranges = self._cached_group_ranges[group_id][:]
            random.shuffle(ranges)

            all_chunks = []
            for start, end in ranges:
                n_cells = end - start
                for chunk_offset in range(0, n_cells, self._chunk_size):
                    chunk_start = start + chunk_offset
                    chunk_end = min(start + chunk_offset + self._chunk_size, end)
                    all_chunks.append((chunk_start, chunk_end))

            random.shuffle(all_chunks)
            leftover = None

            for chunk_start, chunk_end in all_chunks:
                chunk_data = self._load_channels_from_h5(
                    self._h5file, chunk_start, chunk_end
                )

                if leftover is not None:
                    chunk_data = np.concatenate([leftover, chunk_data], axis=0)
                    leftover = None

                n_cells_in_chunk = chunk_data.shape[0]
                n_complete_batches = n_cells_in_chunk // self._cell_batch_size

                for i in range(n_complete_batches):
                    batch_start = i * self._cell_batch_size
                    batch_end = batch_start + self._cell_batch_size
                    batch = chunk_data[batch_start:batch_end]
                    has_spatial = self._n_channels > 1 and bool(batch[:, :, 1:].any())
                    yield group_id, batch, has_spatial

                if n_cells_in_chunk % self._cell_batch_size != 0:
                    leftover = chunk_data[n_complete_batches * self._cell_batch_size :]

            if leftover is not None and leftover.shape[0] > 0:
                needed = self._cell_batch_size - leftover.shape[0]
                if needed > 0 and all_chunks:
                    first_chunk_start, first_chunk_end = all_chunks[0]
                    wraparound = self._load_channels_from_h5(
                        self._h5file,
                        first_chunk_start,
                        min(first_chunk_start + needed, first_chunk_end),
                    )
                    batch = np.concatenate([leftover, wraparound], axis=0)
                    if batch.shape[0] == self._cell_batch_size:
                        has_spatial = self._n_channels > 1 and bool(
                            batch[:, :, 1:].any()
                        )
                        yield group_id, batch, has_spatial

    @classmethod
    def load(cls, path: str | Path, in_memory: bool = False) -> "CogentDataset":
        """Load dataset from HDF5 format."""
        return cls(path=path, in_memory=in_memory)

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self._n_genes

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return self._n_groups

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._n_channels

    @property
    def gene_names(self) -> list[str]:
        """List of gene names."""
        return self._gene_names

    @property
    def group_ids(self) -> list[str]:
        """List of group IDs."""
        return self._group_ids

    def close(self):
        """Close HDF5 file if open."""
        if self._h5file is not None:
            with contextlib.suppress(TypeError, ValueError):
                self._h5file.close()
            self._h5file = None

    def __del__(self):
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            self.close()


class SpatialNeighborCache:
    """Cache for spatial neighbor computations."""

    def __init__(self, radii: Sequence[float]):
        self.radii = list(radii)
        self._trees = {}
        self._neighbors = {}

    def get_or_build_tree(
        self, sample_idx: int, spatial_coords: np.ndarray
    ) -> BallTree:
        """Get cached BallTree or build new one."""
        if sample_idx not in self._trees:
            self._trees[sample_idx] = BallTree(spatial_coords, leaf_size=24)
        return self._trees[sample_idx]

    def compute_all_radii(
        self, sample_idx: int, spatial_coords: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Compute neighbor indices for all radii."""
        tree = self.get_or_build_tree(sample_idx, spatial_coords)
        results = []

        for radius in self.radii:
            key = (sample_idx, radius)
            if key in self._neighbors:
                results.append(self._neighbors[key])
            else:
                ind = tree.query_radius(spatial_coords, r=radius, return_distance=False)
                indices = np.concatenate(ind)
                indptr = np.zeros(len(ind) + 1, dtype=np.int64)
                indptr[1:] = np.cumsum([len(v) for v in ind])
                self._neighbors[key] = (indices, indptr)
                results.append((indices, indptr))

        return results


def compute_spatial_convolution(
    expression: sp.csr_matrix,
    neighbor_indices: np.ndarray,
    neighbor_indptr: np.ndarray,
) -> sp.csr_matrix:
    """Compute spatial convolution given expression and neighbor graph."""
    n_cells = expression.shape[0]

    row_indices, expanded_weights = _build_adjacency_components(neighbor_indptr)

    adjacency = sp.csr_matrix(
        (expanded_weights, (row_indices, neighbor_indices)),
        shape=(n_cells, n_cells),
        dtype=np.float32,
    )

    return adjacency @ expression


class CogentStorageWriter:
    """Writer for creating HDF5 storage files."""

    def __init__(
        self,
        path: Path,
        n_genes: int,
        n_channels: int,
        n_groups: int,
        spatial_conv: list[float] | None = None,
        chunk_size: int = 10000,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.n_genes = n_genes
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.spatial_conv = spatial_conv or []
        self.chunk_size = chunk_size

        self._h5file = None
        self._current_cell = 0
        self._sample_boundaries = [0]

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()
        self.close()

    def _open(self):
        """Open HDF5 file and create structure."""
        h5_path = self.path / "data.h5"
        self._h5file = h5py.File(h5_path, "w")

        self._h5file.attrs["n_genes"] = self.n_genes
        self._h5file.attrs["n_channels"] = self.n_channels
        self._h5file.attrs["n_groups"] = self.n_groups
        if self.spatial_conv:
            self._h5file.attrs["spatial_conv"] = self.spatial_conv

        self._h5file.create_group("expression")
        self._h5file.create_group("spatial")
        self._h5file.create_group("convolved")
        self._h5file.create_group("metadata")

        for c in range(1, self.n_channels):
            self._h5file.create_group(f"convolved/channel_{c}")

        self._expr_data_ds = self._h5file.create_dataset(
            "expression/data",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(self.chunk_size,),
        )
        self._expr_indices_ds = self._h5file.create_dataset(
            "expression/indices",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(self.chunk_size,),
        )
        self._expr_indptr_ds = self._h5file.create_dataset(
            "expression/indptr",
            shape=(1,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(self.chunk_size,),
            data=np.array([0], dtype=np.int64),
        )

        self._conv_data_ds = {}
        self._conv_indices_ds = {}
        self._conv_indptr_ds = {}
        for c in range(1, self.n_channels):
            grp = self._h5file[f"convolved/channel_{c}"]
            self._conv_data_ds[c] = grp.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(self.chunk_size,),
            )
            self._conv_indices_ds[c] = grp.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(self.chunk_size,),
            )
            self._conv_indptr_ds[c] = grp.create_dataset(
                "indptr",
                shape=(1,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(self.chunk_size,),
                data=np.array([0], dtype=np.int64),
            )

        self._spatial_coords_ds = None
        self._cell_to_sample_ds = self._h5file.create_dataset(
            "metadata/cell_to_sample",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(self.chunk_size,),
        )
        self._cell_to_group_ds = self._h5file.create_dataset(
            "metadata/cell_to_group",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(self.chunk_size,),
        )

    def close(self):
        """Close HDF5 file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def add_sample(
        self,
        expression: sp.csr_matrix,
        group_id: int,
        sample_id: int,
        convolved: list[sp.csr_matrix] | None = None,
        spatial_coords: np.ndarray | None = None,
    ):
        """Add a sample's data to storage."""
        n_cells = expression.shape[0]

        expr_csr = expression.tocsr()
        expr_data = expr_csr.data.astype(np.float32)
        expr_indices = expr_csr.indices.astype(np.int32)

        old_size = self._expr_data_ds.shape[0]
        self._expr_data_ds.resize(old_size + len(expr_data), axis=0)
        self._expr_data_ds[old_size:] = expr_data

        old_size = self._expr_indices_ds.shape[0]
        self._expr_indices_ds.resize(old_size + len(expr_indices), axis=0)
        self._expr_indices_ds[old_size:] = expr_indices

        base_offset = self._expr_indptr_ds[-1]
        new_indptr = base_offset + expr_csr.indptr[1:]
        old_size = self._expr_indptr_ds.shape[0]
        self._expr_indptr_ds.resize(old_size + len(new_indptr), axis=0)
        self._expr_indptr_ds[old_size:] = new_indptr

        if convolved is not None:
            for c, conv in enumerate(convolved, start=1):
                conv_csr = conv.tocsr()
                conv_data = conv_csr.data.astype(np.float32)
                conv_indices = conv_csr.indices.astype(np.int32)

                old_size = self._conv_data_ds[c].shape[0]
                self._conv_data_ds[c].resize(old_size + len(conv_data), axis=0)
                self._conv_data_ds[c][old_size:] = conv_data

                old_size = self._conv_indices_ds[c].shape[0]
                self._conv_indices_ds[c].resize(old_size + len(conv_indices), axis=0)
                self._conv_indices_ds[c][old_size:] = conv_indices

                base_offset = self._conv_indptr_ds[c][-1]
                new_indptr = base_offset + conv_csr.indptr[1:]
                old_size = self._conv_indptr_ds[c].shape[0]
                self._conv_indptr_ds[c].resize(old_size + len(new_indptr), axis=0)
                self._conv_indptr_ds[c][old_size:] = new_indptr
        else:
            for c in range(1, self.n_channels):
                base_offset = int(self._conv_indptr_ds[c][-1])
                old_size = self._conv_indptr_ds[c].shape[0]
                self._conv_indptr_ds[c].resize(old_size + n_cells, axis=0)
                self._conv_indptr_ds[c][old_size:] = np.full(
                    n_cells, base_offset, dtype=np.int64
                )

        if spatial_coords is not None:
            coords = spatial_coords.astype(np.float32)
            if self._spatial_coords_ds is None:
                self._spatial_coords_ds = self._h5file.create_dataset(
                    "spatial/coords",
                    shape=(0, coords.shape[1]),
                    maxshape=(None, coords.shape[1]),
                    dtype=np.float32,
                    chunks=(self.chunk_size, coords.shape[1]),
                )
            old_size = self._spatial_coords_ds.shape[0]
            self._spatial_coords_ds.resize(old_size + n_cells, axis=0)
            self._spatial_coords_ds[old_size:] = coords

        cell_to_sample = np.full(n_cells, sample_id, dtype=np.int32)
        old_size = self._cell_to_sample_ds.shape[0]
        self._cell_to_sample_ds.resize(old_size + n_cells, axis=0)
        self._cell_to_sample_ds[old_size:] = cell_to_sample

        cell_to_group = np.full(n_cells, group_id, dtype=np.int32)
        old_size = self._cell_to_group_ds.shape[0]
        self._cell_to_group_ds.resize(old_size + n_cells, axis=0)
        self._cell_to_group_ds[old_size:] = cell_to_group

        self._current_cell += n_cells
        self._sample_boundaries.append(self._current_cell)

    def _finalize(self):
        """Write sample boundaries."""
        self._h5file.create_dataset(
            "metadata/sample_boundaries",
            data=np.array(self._sample_boundaries, dtype=np.int64),
        )


class CogentIterableDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset wrapper for async prefetching.

    Wraps a CogentDataset to enable multi-worker data loading with
    PyTorch DataLoader. Each worker opens its own HDF5 file handle
    for thread safety.

    Args:
        dataset_path: Path to saved CogentDataset directory
        in_memory: Whether to load data into memory (per worker)
    """

    def __init__(
        self,
        dataset_path: Path,
        in_memory: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.in_memory = in_memory
        self._dataset: CogentDataset | None = None

    def _get_dataset(self) -> CogentDataset:
        """Get or create worker-local dataset instance."""
        if self._dataset is None:
            self._dataset = CogentDataset.load(
                self.dataset_path,
                in_memory=self.in_memory,
            )
        return self._dataset

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches, yielding tensors."""
        dataset = self._get_dataset()
        for group_id, expression, has_spatial in dataset:
            yield (
                torch.tensor(group_id, dtype=torch.int64),
                torch.from_numpy(expression).float(),
                torch.tensor(has_spatial, dtype=torch.bool),
            )

    def __del__(self):
        """Cleanup dataset on deletion."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
