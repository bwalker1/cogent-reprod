from pathlib import Path
import numpy as np
from umap import UMAP as _UMAP
from scipy.spatial.distance import cdist
from cogent import Cogent
from cogent.analysis import (
    build_knn_graph,
    compute_leiden_from_connectivity,
    compute_silhouette_score,
    compute_spearman_dist_corr,
    compute_temporal_monotonicity,
    compute_knn_stage_distribution,
    compute_binned_dist_corr_curve,
    compute_stage_pca_embeddings,
    load_zarr_v3_array,
    per_stage_expression_corrs,
)
from cogent.figures.plotting import save_panel


def calculate(data_dir):
    data_dir = Path(data_dir)
    cogent_model_file = data_dir / "mosta/models/cogent/model.pt"
    musegnn_zarr = data_dir / "mosta/models/comparison/musegnn.zarr"
    scetm_zarr = data_dir / "mosta/models/comparison/scetm.zarr"
    h5_e125 = data_dir / "mosta/raw/E12.5_E1S1.MOSTA.h5ad"
    h5_e135 = data_dir / "mosta/raw/E13.5_E1S1.MOSTA.h5ad"
    h5_e145 = data_dir / "mosta/raw/E14.5_E1S1.MOSTA.h5ad"
    h5_e155 = data_dir / "mosta/raw/E15.5_E1S1.MOSTA.h5ad"
    h5_e165 = data_dir / "mosta/raw/E16.5_E1S1.MOSTA.h5ad"
    genes_file = data_dir / "mosta/genes.txt"
    STAGE_FILES = [h5_e125, h5_e135, h5_e145, h5_e155, h5_e165]
    _genes = Path(genes_file).read_text().strip().split("\n")
    _cogent_model = Cogent.load(Path(cogent_model_file).parent, device="cpu")
    _cogent_emb_full = _cogent_model.get_embeddings()
    _cogent_half_dim = _cogent_emb_full.shape[2] // 2
    cogent_embeddings = _cogent_emb_full[:, :, :_cogent_half_dim].astype(np.float32)
    musegnn_embeddings = load_zarr_v3_array(musegnn_zarr, "embeddings").astype(
        np.float32
    )
    scetm_embeddings = load_zarr_v3_array(scetm_zarr, "embeddings").astype(np.float32)
    pca_embeddings = compute_stage_pca_embeddings(STAGE_FILES, _genes)
    cogent_flat = cogent_embeddings.reshape(-1, cogent_embeddings.shape[-1])
    musegnn_flat = musegnn_embeddings.reshape(-1, musegnn_embeddings.shape[-1])
    scetm_flat = scetm_embeddings.reshape(-1, scetm_embeddings.shape[-1])
    pca_flat = pca_embeddings.reshape(-1, pca_embeddings.shape[-1])
    cogent_leiden = compute_leiden_from_connectivity(
        build_knn_graph(cogent_flat, metric="cosine"), random_state=42
    )
    musegnn_leiden = compute_leiden_from_connectivity(
        build_knn_graph(musegnn_flat, metric="euclidean"), random_state=42
    )
    scetm_leiden = compute_leiden_from_connectivity(
        build_knn_graph(scetm_flat, metric="euclidean"), random_state=42
    )
    pca_leiden = compute_leiden_from_connectivity(
        build_knn_graph(pca_flat, metric="euclidean"), random_state=42
    )
    cogent_umap = (
        _UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
        .fit_transform(cogent_flat)
        .astype(np.float32)
    )
    musegnn_umap = (
        _UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
        .fit_transform(musegnn_flat)
        .astype(np.float32)
    )
    scetm_umap = (
        _UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
        .fit_transform(scetm_flat)
        .astype(np.float32)
    )
    pca_umap = (
        _UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
        .fit_transform(pca_flat)
        .astype(np.float32)
    )
    expr_corrs = per_stage_expression_corrs(STAGE_FILES, _genes)
    sil_cogent = float(
        compute_silhouette_score(
            cogent_flat, cogent_leiden, metric="cosine", sample_size=5000
        )
    )
    sil_musegnn = float(
        compute_silhouette_score(
            musegnn_flat, musegnn_leiden, metric="euclidean", sample_size=5000
        )
    )
    sil_scetm = float(
        compute_silhouette_score(
            scetm_flat, scetm_leiden, metric="euclidean", sample_size=5000
        )
    )
    sil_pca = float(
        compute_silhouette_score(
            pca_flat, pca_leiden, metric="euclidean", sample_size=5000
        )
    )
    spearman_cogent = float(compute_spearman_dist_corr(cogent_embeddings, expr_corrs))
    spearman_musegnn = float(compute_spearman_dist_corr(musegnn_embeddings, expr_corrs))
    spearman_scetm = float(compute_spearman_dist_corr(scetm_embeddings, expr_corrs))
    spearman_pca = float(compute_spearman_dist_corr(pca_embeddings, expr_corrs))
    temporal_cogent = float(compute_temporal_monotonicity(cogent_embeddings))
    temporal_musegnn = float(compute_temporal_monotonicity(musegnn_embeddings))
    temporal_scetm = float(compute_temporal_monotonicity(scetm_embeddings))
    temporal_pca = float(compute_temporal_monotonicity(pca_embeddings))

    def compute_trajectory_compactness(embeddings_3d, metric="cosine"):
        same_sum = 0.0
        same_n = 0
        random_sum = 0.0
        random_n = 0
        for i in range(embeddings_3d.shape[0] - 1):
            d = cdist(embeddings_3d[i], embeddings_3d[i + 1], metric=metric).astype(
                np.float32
            )
            same = np.diag(d)
            same_mask = np.isfinite(same)
            same_sum += float(np.sum(same[same_mask]))
            same_n += int(np.sum(same_mask))
            offdiag_mask = ~np.eye(d.shape[0], dtype=bool)
            random = d[offdiag_mask]
            random_mask = np.isfinite(random)
            random_sum += float(np.sum(random[random_mask]))
            random_n += int(np.sum(random_mask))
        same_mean = same_sum / max(same_n, 1)
        random_mean = random_sum / max(random_n, 1)
        return float(1.0 - same_mean / max(random_mean, 1e-09))

    _cogent_stage_ints = np.repeat(
        np.arange(cogent_embeddings.shape[0], dtype=np.int32),
        cogent_embeddings.shape[1],
    )
    _musegnn_stage_ints = np.repeat(
        np.arange(musegnn_embeddings.shape[0], dtype=np.int32),
        musegnn_embeddings.shape[1],
    )
    _scetm_stage_ints = np.repeat(
        np.arange(scetm_embeddings.shape[0], dtype=np.int32), scetm_embeddings.shape[1]
    )
    _pca_stage_ints = np.repeat(
        np.arange(pca_embeddings.shape[0], dtype=np.int32), pca_embeddings.shape[1]
    )
    purity_cogent = float(
        compute_knn_stage_distribution(
            cogent_flat, _cogent_stage_ints, k=20, metric="cosine"
        )["same_stage_frac"]
    )
    purity_musegnn = float(
        compute_knn_stage_distribution(
            musegnn_flat, _musegnn_stage_ints, k=20, metric="euclidean"
        )["same_stage_frac"]
    )
    purity_scetm = float(
        compute_knn_stage_distribution(
            scetm_flat, _scetm_stage_ints, k=20, metric="euclidean"
        )["same_stage_frac"]
    )
    purity_pca = float(
        compute_knn_stage_distribution(
            pca_flat, _pca_stage_ints, k=20, metric="euclidean"
        )["same_stage_frac"]
    )
    compactness_cogent = float(
        compute_trajectory_compactness(cogent_embeddings, metric="cosine")
    )
    compactness_musegnn = float(
        compute_trajectory_compactness(musegnn_embeddings, metric="euclidean")
    )
    compactness_scetm = float(
        compute_trajectory_compactness(scetm_embeddings, metric="euclidean")
    )
    compactness_pca = float(
        compute_trajectory_compactness(pca_embeddings, metric="euclidean")
    )
    curve_cogent = compute_binned_dist_corr_curve(
        cogent_embeddings, expr_corrs, n_bins=20
    )
    curve_musegnn = compute_binned_dist_corr_curve(
        musegnn_embeddings, expr_corrs, n_bins=20
    )
    curve_scetm = compute_binned_dist_corr_curve(
        scetm_embeddings, expr_corrs, n_bins=20
    )
    curve_pca = compute_binned_dist_corr_curve(pca_embeddings, expr_corrs, n_bins=20)
    return {
        "cogent_embeddings": cogent_embeddings,
        "musegnn_embeddings": musegnn_embeddings,
        "scetm_embeddings": scetm_embeddings,
        "pca_embeddings": pca_embeddings,
        "cogent_umap": cogent_umap,
        "musegnn_umap": musegnn_umap,
        "scetm_umap": scetm_umap,
        "pca_umap": pca_umap,
        "sil_cogent": sil_cogent,
        "sil_musegnn": sil_musegnn,
        "sil_scetm": sil_scetm,
        "sil_pca": sil_pca,
        "spearman_cogent": spearman_cogent,
        "spearman_musegnn": spearman_musegnn,
        "spearman_scetm": spearman_scetm,
        "spearman_pca": spearman_pca,
        "temporal_cogent": temporal_cogent,
        "temporal_musegnn": temporal_musegnn,
        "temporal_scetm": temporal_scetm,
        "temporal_pca": temporal_pca,
        "purity_cogent": purity_cogent,
        "purity_musegnn": purity_musegnn,
        "purity_scetm": purity_scetm,
        "purity_pca": purity_pca,
        "compactness_cogent": compactness_cogent,
        "compactness_musegnn": compactness_musegnn,
        "compactness_scetm": compactness_scetm,
        "compactness_pca": compactness_pca,
        "curve_cogent": curve_cogent,
        "curve_musegnn": curve_musegnn,
        "curve_scetm": curve_scetm,
        "curve_pca": curve_pca,
    }


def panel_umap_cogent(cogent_embeddings, cogent_umap):
    coords = cogent_umap
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    n_stages = cogent_embeddings.shape[0]
    n_genes = cogent_embeddings.shape[1]
    STAGE_LABELS_LOCAL = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    values = [STAGE_LABELS_LOCAL[s] for s in range(n_stages) for _ in range(n_genes)]
    type = "categorical"
    return {"x": x, "y": y, "values": values, "type": type}


def panel_umap_musegnn(musegnn_embeddings, musegnn_umap):
    coords = musegnn_umap
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    n_stages = musegnn_embeddings.shape[0]
    n_genes = musegnn_embeddings.shape[1]
    STAGE_LABELS_LOCAL = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    values = [STAGE_LABELS_LOCAL[s] for s in range(n_stages) for _ in range(n_genes)]
    type = "categorical"
    return {"x": x, "y": y, "values": values, "type": type}


def panel_umap_scetm(scetm_embeddings, scetm_umap):
    coords = scetm_umap
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    n_stages = scetm_embeddings.shape[0]
    n_genes = scetm_embeddings.shape[1]
    STAGE_LABELS_LOCAL = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    values = [STAGE_LABELS_LOCAL[s] for s in range(n_stages) for _ in range(n_genes)]
    type = "categorical"
    return {"x": x, "y": y, "values": values, "type": type}


def panel_umap_pca(pca_embeddings, pca_umap):
    coords = pca_umap
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    n_stages = pca_embeddings.shape[0]
    n_genes = pca_embeddings.shape[1]
    STAGE_LABELS_LOCAL = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    values = [STAGE_LABELS_LOCAL[s] for s in range(n_stages) for _ in range(n_genes)]
    type = "categorical"
    return {"x": x, "y": y, "values": values, "type": type}


def panel_bar_silhouette(sil_cogent, sil_musegnn, sil_scetm, sil_pca):
    return {
        "bars": [
            {"label": "Cogent", "value": sil_cogent},
            {"label": "MuSe-GNN", "value": sil_musegnn},
            {"label": "scETM", "value": sil_scetm},
            {"label": "PCA", "value": sil_pca},
        ]
    }


def panel_bar_spearman(spearman_cogent, spearman_musegnn, spearman_scetm, spearman_pca):
    return {
        "bars": [
            {"label": "Cogent", "value": spearman_cogent},
            {"label": "MuSe-GNN", "value": spearman_musegnn},
            {"label": "scETM", "value": spearman_scetm},
            {"label": "PCA", "value": spearman_pca},
        ]
    }


def panel_bar_stage_purity(purity_cogent, purity_musegnn, purity_scetm, purity_pca):
    return {
        "bars": [
            {"label": "Cogent", "value": purity_cogent},
            {"label": "MuSe-GNN", "value": purity_musegnn},
            {"label": "scETM", "value": purity_scetm},
            {"label": "PCA", "value": purity_pca},
        ]
    }


def panel_bar_temporal(temporal_cogent, temporal_musegnn, temporal_scetm, temporal_pca):
    return {
        "bars": [
            {"label": "Cogent", "value": temporal_cogent},
            {"label": "MuSe-GNN", "value": temporal_musegnn},
            {"label": "scETM", "value": temporal_scetm},
            {"label": "PCA", "value": temporal_pca},
        ]
    }


def panel_bar_compactness(
    compactness_cogent, compactness_musegnn, compactness_scetm, compactness_pca
):
    return {
        "bars": [
            {"label": "Cogent", "value": compactness_cogent},
            {"label": "MuSe-GNN", "value": compactness_musegnn},
            {"label": "scETM", "value": compactness_scetm},
            {"label": "PCA", "value": compactness_pca},
        ]
    }


def panel_hist_cogent(curve_cogent):
    c = curve_cogent
    points = [
        {"x": float(c["x"][i]), "y": float(c["y"][i])} for i in range(len(c["x"]))
    ]
    series = [{"name": "Cogent", "points": points, "color": "#f97316"}]
    return {"series": series}


def panel_hist_musegnn(curve_musegnn):
    c = curve_musegnn
    points = [
        {"x": float(c["x"][i]), "y": float(c["y"][i])} for i in range(len(c["x"]))
    ]
    series = [{"name": "MuSe-GNN", "points": points, "color": "#0ea5e9"}]
    return {"series": series}


def panel_hist_scetm(curve_scetm):
    c = curve_scetm
    points = [
        {"x": float(c["x"][i]), "y": float(c["y"][i])} for i in range(len(c["x"]))
    ]
    series = [{"name": "scETM", "points": points, "color": "#22c55e"}]
    return {"series": series}


def panel_hist_pca(curve_pca):
    c = curve_pca
    points = [
        {"x": float(c["x"][i]), "y": float(c["y"][i])} for i in range(len(c["x"]))
    ]
    series = [{"name": "PCA", "points": points, "color": "#ec4899"}]
    return {"series": series}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_umap_cogent(
            cogent_embeddings=results["cogent_embeddings"],
            cogent_umap=results["cogent_umap"],
        ),
        "umap",
        output_dir / "umap-cogent",
        "Cogent",
        {},
    )
    save_panel(
        panel_umap_musegnn(
            musegnn_embeddings=results["musegnn_embeddings"],
            musegnn_umap=results["musegnn_umap"],
        ),
        "umap",
        output_dir / "umap-musegnn",
        "MuSe-GNN",
        {},
    )
    save_panel(
        panel_umap_scetm(
            scetm_embeddings=results["scetm_embeddings"],
            scetm_umap=results["scetm_umap"],
        ),
        "umap",
        output_dir / "umap-scetm",
        "scETM",
        {},
    )
    save_panel(
        panel_umap_pca(
            pca_embeddings=results["pca_embeddings"], pca_umap=results["pca_umap"]
        ),
        "umap",
        output_dir / "umap-pca",
        "PCA",
        {},
    )
    save_panel(
        panel_bar_silhouette(
            sil_cogent=results["sil_cogent"],
            sil_musegnn=results["sil_musegnn"],
            sil_scetm=results["sil_scetm"],
            sil_pca=results["sil_pca"],
        ),
        "bar",
        output_dir / "bar-silhouette",
        "Cluster separation",
        {"ylabel": "Silhouette score"},
    )
    save_panel(
        panel_bar_spearman(
            spearman_cogent=results["spearman_cogent"],
            spearman_musegnn=results["spearman_musegnn"],
            spearman_scetm=results["spearman_scetm"],
            spearman_pca=results["spearman_pca"],
        ),
        "bar",
        output_dir / "bar-spearman",
        "Distance–correlation",
        {"ylabel": "Spearman ρ (dist vs corr)", "ymax": 0, "ymin": -0.7},
    )
    save_panel(
        panel_bar_stage_purity(
            purity_cogent=results["purity_cogent"],
            purity_musegnn=results["purity_musegnn"],
            purity_scetm=results["purity_scetm"],
            purity_pca=results["purity_pca"],
        ),
        "bar",
        output_dir / "bar-stage-purity",
        "Stage isolation",
        {"ylabel": "Same-stage neighbors (k=20)", "ymax": 0.9, "ymin": 0},
    )
    save_panel(
        panel_bar_temporal(
            temporal_cogent=results["temporal_cogent"],
            temporal_musegnn=results["temporal_musegnn"],
            temporal_scetm=results["temporal_scetm"],
            temporal_pca=results["temporal_pca"],
        ),
        "bar",
        output_dir / "bar-temporal",
        "Temporal monotonicity",
        {"ylabel": "Spearman ρ", "ymax": 0.3, "ymin": 0},
    )
    save_panel(
        panel_bar_compactness(
            compactness_cogent=results["compactness_cogent"],
            compactness_musegnn=results["compactness_musegnn"],
            compactness_scetm=results["compactness_scetm"],
            compactness_pca=results["compactness_pca"],
        ),
        "bar",
        output_dir / "bar-compactness",
        "Compactness",
        {"ylabel": "Compactness", "ymax": 1, "ymin": 0},
    )
    save_panel(
        panel_hist_cogent(curve_cogent=results["curve_cogent"]),
        "line",
        output_dir / "hist-cogent",
        "Cogent",
        {
            "step": True,
            "xlabel": "Distance percentile",
            "ylabel": "Expression correlation",
            "ymax": 0.12,
            "ymin": -0.05,
        },
    )
    save_panel(
        panel_hist_musegnn(curve_musegnn=results["curve_musegnn"]),
        "line",
        output_dir / "hist-musegnn",
        "MuSe-GNN",
        {
            "step": True,
            "xlabel": "Distance percentile",
            "ylabel": "Expression correlation",
            "ymax": 0.12,
            "ymin": -0.05,
        },
    )
    save_panel(
        panel_hist_scetm(curve_scetm=results["curve_scetm"]),
        "line",
        output_dir / "hist-scetm",
        "scETM",
        {
            "step": True,
            "xlabel": "Distance percentile",
            "ylabel": "Expression correlation",
            "ymax": 0.12,
            "ymin": -0.05,
        },
    )
    save_panel(
        panel_hist_pca(curve_pca=results["curve_pca"]),
        "line",
        output_dir / "hist-pca",
        "PCA",
        {
            "step": True,
            "xlabel": "Distance percentile",
            "ylabel": "Expression correlation",
            "ymax": 0.12,
            "ymin": -0.05,
        },
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "fig3")
