"""Gene embedding and spatial expression analyses."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    embeddings: NDArray[np.float32],
    n_neighbors: int = 15,
    metric: str = "cosine",
    approximate: bool | None = None,
    random_state: int = 42,
) -> sparse.csr_matrix:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return sparse.csr_matrix((n_samples, n_samples), dtype=np.float32)
    n_neighbors = min(n_neighbors, n_samples - 1)

    if approximate is None:
        approximate = n_samples > 4096

    if approximate:
        from pynndescent import NNDescent

        index = NNDescent(
            embeddings,
            n_neighbors=n_neighbors + 1,
            metric=metric,
            random_state=random_state,
            n_jobs=-1,
        )
        indices, distances = index.neighbor_graph
    else:
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric=metric,
            algorithm="auto",
        ).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

    rows = np.repeat(np.arange(n_samples, dtype=np.int32), n_neighbors)
    cols = indices[:, 1 : n_neighbors + 1].reshape(-1).astype(np.int32)
    neighbor_distances = distances[:, 1 : n_neighbors + 1].reshape(-1)

    if metric == "cosine":
        weights = 1.0 - neighbor_distances
    else:
        scale = float(np.median(distances[:, 1:]))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        weights = np.exp(-neighbor_distances / scale)

    connectivity = sparse.csr_matrix(
        (weights.astype(np.float32), (rows, cols)),
        shape=(n_samples, n_samples),
    )
    return connectivity.maximum(connectivity.T).tocsr()


def compute_leiden_from_connectivity(
    connectivity: NDArray[np.float32] | sparse.spmatrix,
    resolution: float = 1.0,
    random_state: int | None = None,
) -> NDArray[np.int32]:
    n_samples = connectivity.shape[0]

    if sparse.issparse(connectivity):
        coo = sparse.triu(connectivity, k=1).tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist(), strict=True))
        weights = coo.data.astype(float).tolist()
    else:
        rows, cols = np.triu_indices(n_samples, k=1)
        mask = np.asarray(connectivity)[rows, cols] > 0
        edges = list(zip(rows[mask].tolist(), cols[mask].tolist(), strict=True))
        weights = (
            np.asarray(connectivity)[rows[mask], cols[mask]].astype(float).tolist()
        )

    graph = ig.Graph(n=n_samples, edges=edges, directed=False)
    graph.es["weight"] = weights

    if random_state is not None:
        import random

        ig.set_random_number_generator(random.Random(random_state))

    partition = graph.community_leiden(
        weights="weight",
        resolution=resolution,
        objective_function="modularity",
        n_iterations=10,
    )
    return np.asarray(partition.membership, dtype=np.int32)


def compute_silhouette_score(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    metric: str = "cosine",
    sample_size: int | None = None,
    random_state: int | None = 42,
) -> float:
    labels = np.asarray(labels)
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return 0.0

    kwargs = {}
    if sample_size is not None and sample_size < len(labels):
        kwargs["sample_size"] = sample_size
        kwargs["random_state"] = random_state

    return float(silhouette_score(embeddings, labels, metric=metric, **kwargs))


def load_zarr_v3_array(store_path: str | Path, array_name: str) -> NDArray[np.float32]:
    import zarr

    return np.asarray(zarr.open(str(store_path), mode="r")[array_name][:])


def pooled_dist_corr_pairs(
    embeddings_3d: NDArray[np.float32],
    expr_corrs_per_stage: list[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    all_distances = []
    all_correlations = []
    for i in range(embeddings_3d.shape[0]):
        distances = pdist(embeddings_3d[i], metric="cosine").astype(np.float32)
        correlations = expr_corrs_per_stage[i]
        mask = ~(np.isnan(distances) | np.isnan(correlations))
        all_distances.append(distances[mask])
        all_correlations.append(correlations[mask])
    return np.concatenate(all_distances), np.concatenate(all_correlations)


def compute_spearman_dist_corr(
    embeddings_3d: NDArray[np.float32],
    expr_corrs_per_stage: list[NDArray[np.float32]],
    subsample: int = 1_000_000,
    random_state: int = 42,
) -> float:
    distances, correlations = pooled_dist_corr_pairs(
        embeddings_3d, expr_corrs_per_stage
    )
    if len(distances) > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(distances), subsample, replace=False)
        distances = distances[idx]
        correlations = correlations[idx]
    return float(spearmanr(distances, correlations).correlation)


def compute_temporal_monotonicity(
    embeddings_3d: NDArray[np.float32],
) -> float:
    n_stages = embeddings_3d.shape[0]
    gaps = np.asarray(
        [j - i for i in range(n_stages) for j in range(i + 1, n_stages)],
        dtype=np.float32,
    )
    rhos = np.empty(embeddings_3d.shape[1], dtype=np.float32)
    for gene_idx in range(embeddings_3d.shape[1]):
        distances = pdist(embeddings_3d[:, gene_idx, :], metric="cosine")
        rhos[gene_idx] = spearmanr(distances, gaps).correlation
    return float(np.nanmean(rhos))


def compute_knn_stage_distribution(
    flat_embeddings: NDArray[np.float32],
    stage_labels: NDArray[np.int32],
    k: int = 20,
    metric: str = "cosine",
) -> dict[str, float]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(flat_embeddings)
    _, indices = nbrs.kneighbors(flat_embeddings)
    neighbor_stages = stage_labels[indices[:, 1:]]
    same_stage_frac = float((neighbor_stages == stage_labels[:, None]).mean())

    n_stages = int(stage_labels.max()) + 1
    counts = np.zeros((len(stage_labels), n_stages), dtype=np.int32)
    for stage in range(n_stages):
        counts[:, stage] = (neighbor_stages == stage).sum(axis=1)
    probs = counts / k
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    mean_entropy = float(entropy.mean())
    return {
        "same_stage_frac": same_stage_frac,
        "mean_entropy": mean_entropy,
        "entropy_normalized": mean_entropy / np.log(n_stages),
    }


def compute_binned_dist_corr_curve(
    embeddings_3d: NDArray[np.float32],
    expr_corrs_per_stage: list[NDArray[np.float32]],
    n_bins: int = 40,
    subsample: int = 500_000,
    random_state: int = 42,
) -> dict[str, list[float]]:
    distances, correlations = pooled_dist_corr_pairs(
        embeddings_3d, expr_corrs_per_stage
    )
    if len(distances) > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(distances), subsample, replace=False)
        distances = distances[idx]
        correlations = correlations[idx]

    order = np.argsort(distances, kind="stable")
    sorted_correlations = correlations[order]
    n_pairs = len(sorted_correlations)
    edges = np.linspace(0, n_pairs, n_bins + 1).astype(np.int64)

    xs = []
    ys = []
    errs = []
    for bin_idx in range(n_bins):
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        if hi <= lo:
            continue
        segment = sorted_correlations[lo:hi]
        xs.append(0.5 * (lo + hi) / n_pairs)
        ys.append(float(segment.mean()))
        errs.append(float(segment.std()))

    return {"x": xs, "y": ys, "error": errs}


def compute_cluster_sample_scores(
    model,
    samples: Iterable[tuple[str | Path, int]],
    preproc_dir: str | Path | None = None,
    resolution: float = 0.5,
    zone_obs: str = "inj_zone",
    inj_zone_value: str = "wound",
    uninj_zone_value: str = "uninj",
):
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse

    preproc_dir_path = Path(preproc_dir) if preproc_dir is not None else None
    gene_names = list(model.gene_names)
    labels = model._leiden_results[resolution]["labels"]["average"]
    cluster_genes = {
        int(cluster): [
            gene_names[i] for i, label in enumerate(labels) if label == cluster
        ]
        for cluster in sorted(np.unique(labels))
    }

    rows = []
    for sample, dpi in samples:
        sample_str = str(sample)
        if "/" in sample_str or sample_str.endswith(".h5ad"):
            path = Path(sample_str)
            sample_name = path.stem
        else:
            if preproc_dir_path is None:
                raise ValueError(
                    f"sample {sample_str!r} is a bare name but preproc_dir is None"
                )
            path = preproc_dir_path / f"{sample_str}.h5ad"
            sample_name = sample_str

        adata = sc.read_h5ad(path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        var_idx = {gene: i for i, gene in enumerate(adata.var_names)}

        zone = adata.obs[zone_obs].astype(str).values
        masks = {
            "inj": zone == inj_zone_value,
            "uninj": zone == uninj_zone_value,
        }

        for cluster, genes in cluster_genes.items():
            idxs = [var_idx[gene] for gene in genes if gene in var_idx]
            if not idxs:
                continue
            sub = adata.X[:, idxs]
            per_cell = (
                np.asarray(sub.mean(axis=1)).ravel()
                if issparse(sub)
                else sub.mean(axis=1)
            )
            for partition, mask in masks.items():
                n_cells = int(mask.sum())
                if n_cells == 0:
                    continue
                rows.append(
                    {
                        "sample": sample_name,
                        "dpi": dpi,
                        "cluster": cluster,
                        "inj_uninj": partition,
                        "n_cells": n_cells,
                        "mean_score": float(per_cell[mask].mean()),
                    }
                )

    return pd.DataFrame(rows)


def compute_cluster_stage_shifts(model, resolution: float = 0.5):
    import pandas as pd

    embeddings = model._embeddings
    if embeddings is None:
        raise ValueError("Model has no embeddings.")
    half = embeddings.shape[-1] // 2
    emb_mag = embeddings[..., :half]

    labels = np.asarray(
        model._leiden_results[resolution]["labels"]["average"],
        dtype=int,
    )
    group_ids = list(model.group_ids or [])
    if len(group_ids) < 2:
        raise ValueError("Need at least two groups to compute stage shifts.")

    transitions = [(group_ids[i], group_ids[i + 1]) for i in range(len(group_ids) - 1)]
    col_names = [f"{a}→{b}" for a, b in transitions]

    rows = []
    for cluster in sorted(np.unique(labels)):
        gene_mask = labels == cluster
        n_genes = int(gene_mask.sum())
        if n_genes == 0:
            continue
        centroids = emb_mag[:, gene_mask, :].mean(axis=1)
        norms = np.linalg.norm(centroids, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        unit = centroids / norms[:, None]
        distances = [
            float(1.0 - np.dot(unit[i], unit[i + 1])) for i in range(len(group_ids) - 1)
        ]
        rows.append(
            {
                "cluster": int(cluster),
                "n_genes": n_genes,
                **dict(zip(col_names, distances, strict=True)),
                "mean_shift": float(np.mean(distances)),
            }
        )

    return pd.DataFrame(rows).set_index("cluster")


def compute_gene_similarity_matrix(model, resolution: float = 0.5):
    embeddings = model._embeddings
    if embeddings is None:
        raise ValueError("Model has no embeddings.")
    half = embeddings.shape[-1] // 2
    avg = embeddings[..., :half].mean(axis=0)

    labels = np.asarray(
        model._leiden_results[resolution]["labels"]["average"],
        dtype=int,
    )
    gene_names = list(model.gene_names)

    from scipy.cluster.hierarchy import leaves_list, linkage

    def leaf_order(matrix: NDArray[np.float32]) -> NDArray[np.int64]:
        unit = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        linkage_matrix = linkage(unit, method="average", metric="cosine")
        return leaves_list(linkage_matrix).astype(np.int64)

    unique_clusters = np.unique(labels)
    if len(unique_clusters) >= 2:
        centroids = np.stack(
            [avg[labels == cluster].mean(axis=0) for cluster in unique_clusters]
        )
        ordered_clusters = unique_clusters[leaf_order(centroids)]
    else:
        ordered_clusters = unique_clusters

    order_parts = []
    for cluster in ordered_clusters:
        idx = np.where(labels == cluster)[0]
        if len(idx) >= 3:
            idx = idx[leaf_order(avg[idx])]
        order_parts.append(idx)
    order = (
        np.concatenate(order_parts) if order_parts else np.asarray([], dtype=np.int64)
    )

    sub = avg[order]
    unit = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
    sim_mat = (unit @ unit.T).astype(np.float32)
    return {
        "values": sim_mat,
        "genes": [gene_names[i] for i in order],
        "clusters": [int(labels[i]) for i in order],
    }


def compute_stage_pca_embeddings(
    stage_files: Iterable[str | Path],
    genes: list[str],
    n_components: int = 64,
    random_state: int = 42,
) -> NDArray[np.float32]:
    import scanpy as sc
    from sklearn.decomposition import PCA, TruncatedSVD

    per_stage = []
    for path in stage_files:
        adata = sc.read_h5ad(path, backed="r")
        try:
            gene_idx = {gene: i for i, gene in enumerate(adata.var_names)}
            ordered_idx = [gene_idx[gene] for gene in genes if gene in gene_idx]
            matrix = adata.X[:, ordered_idx].T
            if sparse.issparse(matrix):
                pca = TruncatedSVD(n_components=n_components, random_state=random_state)
                embedding = pca.fit_transform(matrix)
            else:
                pca = PCA(n_components=n_components, random_state=random_state)
                embedding = pca.fit_transform(np.asarray(matrix))
        finally:
            adata.file.close()
        embedding = embedding / (
            np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        )
        per_stage.append(embedding.astype(np.float32))

    return np.stack(per_stage)


def per_stage_expression_corrs(
    stage_files: Iterable[str | Path],
    genes: list[str],
) -> list[NDArray[np.float32]]:
    import scanpy as sc

    corrs = []
    for path in stage_files:
        adata = sc.read_h5ad(path)
        gene_idx = {gene: i for i, gene in enumerate(adata.var_names)}
        keep_idx = [gene_idx[gene] for gene in genes if gene in gene_idx]
        matrix = adata.X[:, keep_idx]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        corr = np.corrcoef(matrix.T.astype(np.float32))
        upper = np.triu_indices(corr.shape[0], k=1)
        corrs.append(corr[upper].astype(np.float32))

    return corrs


__all__ = [
    "build_knn_graph",
    "compute_binned_dist_corr_curve",
    "compute_cluster_sample_scores",
    "compute_cluster_stage_shifts",
    "compute_gene_similarity_matrix",
    "compute_knn_stage_distribution",
    "compute_leiden_from_connectivity",
    "compute_silhouette_score",
    "compute_spearman_dist_corr",
    "compute_stage_pca_embeddings",
    "compute_temporal_monotonicity",
    "load_zarr_v3_array",
    "per_stage_expression_corrs",
    "pooled_dist_corr_pairs",
]
