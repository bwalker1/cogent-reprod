"""Gene embedding and module expression utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import anndata as ad

    from .cogent import Cogent


def cosine_distance(
    u: torch.Tensor,
    v: torch.Tensor,
    expand: bool = True,
) -> torch.Tensor:
    if expand:
        return 1 - torch.einsum(
            "bhd,ghd->bhg",
            F.normalize(u, dim=-1),
            F.normalize(v, dim=-1),
        )
    return 1 - torch.sum(u * v, dim=-1)


def cluster_means_all(
    model: Cogent,
    adata: ad.AnnData,
    *,
    view: str | int = "average",
    resolution: float,
) -> dict[int, np.ndarray]:
    if resolution not in model._leiden_results:
        available = sorted(model._leiden_results.keys())
        raise ValueError(
            f"No precomputed Leiden clustering for resolution={resolution}. "
            f"Available: {available}"
        )

    leiden_labels = model._leiden_results[resolution]["labels"]
    if view not in leiden_labels:
        raise ValueError(
            f"View {view!r} not found in Leiden labels. "
            f"Available: {sorted(leiden_labels.keys())}"
        )

    if model._gene_names is None:
        raise ValueError("Model has no gene names.")

    labels = np.asarray(leiden_labels[view], dtype=int)
    gene_names = np.asarray(model._gene_names, dtype=object)
    var_names = set(map(str, adata.var_names))

    out: dict[int, np.ndarray] = {}
    for cluster_id in np.unique(labels):
        cluster_genes = [
            str(gene)
            for gene in gene_names[labels == cluster_id]
            if str(gene) in var_names
        ]
        if not cluster_genes:
            continue
        sub = adata[:, cluster_genes].X
        if sp.issparse(sub):
            mean = np.asarray(sub.mean(axis=1)).ravel()
        else:
            mean = np.asarray(sub).mean(axis=1)
        out[int(cluster_id)] = mean.astype(np.float64, copy=False)

    return out


__all__ = [
    "cluster_means_all",
    "cosine_distance",
]


def normalize_expression(X: sp.csr_matrix, normalize: bool) -> sp.csr_matrix:
    """Normalize expression data.

    Args:
        X: CSR sparse matrix (n_cells, n_genes)
        normalize: Whether to apply log1p normalization

    Returns:
        Normalized CSR sparse matrix
    """
    if normalize:
        row_sums = np.array(X.sum(axis=1)).flatten() + 1e-12
        X = X.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()
        X.data = np.log1p(X.data)

    col_min = np.array(X.min(axis=0).toarray()).flatten()
    X = X.tocoo()
    X.data = X.data - col_min[X.col]
    X = X.tocsr()

    col_max = np.array(X.max(axis=0).toarray()).flatten() + 1e-12
    X = X.multiply(1.0 / col_max).tocsr()

    return X
