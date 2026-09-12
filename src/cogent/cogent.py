"""Cogent models and saved embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cogent.model import Model


class Cogent:
    """Loaded Cogent checkpoint with embedding accessors."""

    def __init__(
        self,
        embedding_dim: int = 64,
        n_heads: int = 1,
        decoder_depth: int = 4,
        cross_p: float = 0.05,
        gene_batch_size: int = 512,
        beta: float = 5.0,
        emb_dropout: float = 0.0,
        decoder_dim: int = 64,
        rand_embeddings: bool = False,
        lambda_reg: float = 0.001,
    ):
        self._embedding_dim = embedding_dim
        self._n_heads = n_heads
        self._decoder_depth = decoder_depth
        self._cross_p = cross_p
        self._gene_batch_size = gene_batch_size
        self._beta = beta
        self._emb_dropout = emb_dropout
        self._decoder_dim = decoder_dim
        self._rand_embeddings = rand_embeddings
        self._lambda_reg = lambda_reg

        self._model: Model | None = None
        self._embeddings: np.ndarray | None = None
        self._gene_names: list[str] | None = None
        self._group_ids: list | None = None
        self._umap_coords: dict | None = None
        self._leiden_results: dict[float, dict] = {}

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Cogent:
        """Load a Cogent checkpoint directory containing ``model.pt``."""
        checkpoint = torch.load(
            Path(path) / "model.pt",
            map_location=device,
            weights_only=False,
        )
        hyperparams = checkpoint["hyperparameters"]
        metadata = checkpoint["metadata"]
        geometry = hyperparams.get("geometry", "cosine")
        if geometry != "cosine":
            raise ValueError(
                f"Cogent manuscript figures use cosine checkpoints; got geometry={geometry!r}."
            )
        cogent = cls(
            embedding_dim=hyperparams["embedding_dim"],
            n_heads=hyperparams["n_heads"],
            decoder_depth=hyperparams["decoder_depth"],
            cross_p=hyperparams["cross_p"],
            gene_batch_size=hyperparams["gene_batch_size"],
            beta=hyperparams["beta"],
            emb_dropout=hyperparams["emb_dropout"],
            decoder_dim=hyperparams["decoder_dim"],
            rand_embeddings=hyperparams["rand_embeddings"],
            lambda_reg=hyperparams.get("lambda_reg", 0.001),
        )

        model = Model(
            n_genes=metadata["n_genes"],
            n_groups=metadata["n_groups"],
            n_channels=metadata["n_channels"],
            embedding_dim=hyperparams["embedding_dim"],
            n_heads=hyperparams["n_heads"],
            decoder_depth=hyperparams["decoder_depth"],
            cross_p=hyperparams["cross_p"],
            gene_batch_size=hyperparams["gene_batch_size"],
            beta=hyperparams["beta"],
            emb_dropout=hyperparams["emb_dropout"],
            decoder_dim=hyperparams["decoder_dim"],
            rand_embeddings=hyperparams["rand_embeddings"],
            lambda_reg=hyperparams.get("lambda_reg", 0.001),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        cogent._model = model
        cogent._gene_names = list(metadata["gene_names"])
        cogent._group_ids = list(metadata["group_ids"])
        cogent._umap_coords = checkpoint.get("umap_coords")
        cogent._leiden_results = checkpoint.get("leiden_results") or {}

        embeddings = checkpoint.get("embeddings")
        if embeddings is None:
            embeddings = model.embeddings.detach().cpu().numpy()
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        cogent._embeddings = np.asarray(embeddings)

        return cogent

    def get_embeddings(self) -> np.ndarray:
        """Return interaction, spatial-range, and decoder embedding parameters.

        The first model.embedding_dim coordinates define gene interactions.
        """
        if self._embeddings is None:
            if self._model is None:
                raise ValueError("No checkpoint has been loaded.")
            self._embeddings = self._model.embeddings.detach().cpu().numpy()
        return self._embeddings

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def gene_names(self) -> list[str] | None:
        return self._gene_names

    @property
    def group_ids(self) -> list | None:
        return self._group_ids

    def compute_umap(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int | None = None,
        shared_n_neighbors: int = 100,
        intermediate_dim: int = 16,
    ) -> dict:
        """Compute UMAP projections for each group and shared embeddings.

        Uses a two-stage UMAP process:
        1. Reduce embeddings to intermediate_dim (default 16) for clustering
        2. Reduce to 2D for visualization

        Also computes two shared UMAPs:
        - 'shared': concatenated embeddings from all groups
        - 'average': averaged distance matrix across groups

        Results are cached in self._umap_coords (2D) and self._umap_intermediate (16D).

        Args:
            n_neighbors: Number of neighbors for per-group UMAP
            min_dist: Minimum distance parameter for UMAP
            metric: Distance metric ('cosine' or 'euclidean')
            random_state: Random seed
            shared_n_neighbors: Number of neighbors for shared UMAPs (default: 100)
            intermediate_dim: Dimension for intermediate UMAP (default: 16)

        Returns:
            Dict mapping group_id to UMAP coordinates (n_genes, 2),
            plus 'shared' (n_groups * n_genes, 2) and 'average' (n_genes, 2)
        """
        from scipy.spatial.distance import pdist, squareform
        from umap import UMAP
        from cogent.analysis import build_knn_graph

        if self._embeddings is None:
            raise ValueError("No embeddings available. Train model first.")
        embeddings = self._embeddings
        (n_groups, n_genes, embedding_dim) = embeddings.shape
        half_dim = embedding_dim // 2
        embeddings_dist = embeddings[..., :half_dim]
        umap_coords = {}
        umap_intermediate = {}
        connectivities = {}
        distance_matrices = []
        for group_id in range(n_groups):
            emb = embeddings_dist[group_id]
            umap_stage1 = UMAP(
                n_components=intermediate_dim,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )
            coords_intermediate = umap_stage1.fit_transform(emb).astype(np.float32)
            umap_intermediate[group_id] = coords_intermediate
            umap_stage2 = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=random_state,
            )
            coords_2d = umap_stage2.fit_transform(coords_intermediate).astype(
                np.float32
            )
            umap_coords[group_id] = coords_2d
            connectivity = build_knn_graph(
                coords_intermediate, n_neighbors=n_neighbors, metric="euclidean"
            )
            connectivities[group_id] = connectivity
            dist_matrix = squareform(pdist(emb, metric=metric))
            distance_matrices.append(dist_matrix)
        embeddings_flat = embeddings_dist.reshape(-1, half_dim)
        umap_shared_stage1 = UMAP(
            n_components=intermediate_dim,
            n_neighbors=shared_n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        shared_intermediate = umap_shared_stage1.fit_transform(embeddings_flat).astype(
            np.float32
        )
        umap_intermediate["shared"] = shared_intermediate
        umap_shared_stage2 = UMAP(
            n_components=2,
            n_neighbors=shared_n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        shared_coords = umap_shared_stage2.fit_transform(shared_intermediate).astype(
            np.float32
        )
        umap_coords["shared"] = shared_coords
        avg_dist_matrix = np.mean(np.array(distance_matrices), axis=0)
        umap_avg = UMAP(
            n_components=2,
            n_neighbors=shared_n_neighbors,
            min_dist=min_dist,
            metric="precomputed",
            random_state=random_state,
        )
        avg_coords = umap_avg.fit_transform(avg_dist_matrix).astype(np.float32)
        umap_coords["average"] = avg_coords
        self._umap_coords = umap_coords
        self._umap_intermediate = umap_intermediate
        self._connectivities = connectivities
        return umap_coords

    def compute_leiden(
        self,
        resolution: float = 1.0,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        random_state: int | None = 0,
    ) -> dict:
        """Compute Leiden clustering for each group.

        If intermediate UMAP coordinates (16D) are available from compute_umap(),
        clusters on those for consistency with visualization while preserving
        more information than 2D. Otherwise builds k-NN graphs from embeddings.
        Results are cached in self._leiden_labels.

        Args:
            resolution: Resolution parameter for Leiden
            n_neighbors: Number of neighbors for k-NN graph
            metric: Distance metric (default 'euclidean' for UMAP space)
            random_state: Random seed

        Returns:
            Dict mapping group_id to cluster labels (n_genes,)
        """
        from cogent.analysis import build_knn_graph, compute_leiden_from_connectivity

        if self._embeddings is None:
            raise ValueError("No embeddings available. Train model first.")
        embeddings = self._embeddings
        (n_groups, n_genes, embedding_dim) = embeddings.shape
        use_intermediate = (
            hasattr(self, "_umap_intermediate") and self._umap_intermediate is not None
        )
        leiden_labels = {}
        connectivities = {}
        for group_id in range(n_groups):
            if use_intermediate:
                emb = self._umap_intermediate[group_id]
            else:
                half_dim = embedding_dim // 2
                emb = embeddings[group_id, :, :half_dim]
            connectivity = build_knn_graph(emb, n_neighbors=n_neighbors, metric=metric)
            connectivities[group_id] = connectivity
            labels = compute_leiden_from_connectivity(
                connectivity, resolution=resolution, random_state=random_state
            )
            leiden_labels[group_id] = labels
        if use_intermediate:
            avg_intermediate = np.mean(
                np.stack([self._umap_intermediate[g] for g in range(n_groups)]), axis=0
            )
        else:
            half_dim = embedding_dim // 2
            avg_intermediate = embeddings[:, :, :half_dim].mean(axis=0)
        avg_connectivity = build_knn_graph(
            avg_intermediate, n_neighbors=n_neighbors, metric=metric
        )
        connectivities["average"] = avg_connectivity
        leiden_labels["average"] = compute_leiden_from_connectivity(
            avg_connectivity, resolution=resolution, random_state=random_state
        )
        self._leiden_results[resolution] = {
            "labels": leiden_labels,
            "connectivities": connectivities,
        }
        self._leiden_labels = leiden_labels
        self._connectivities = connectivities
        return leiden_labels
