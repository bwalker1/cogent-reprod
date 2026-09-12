from pathlib import Path
from cogent import Cogent
from cogent.utils import cluster_means_all
from cogent.analysis import compute_cluster_sample_scores
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(data_dir, dataset="artista/preprocessed/2DPI_1.h5ad", resolution=1.0):
    data_dir = Path(data_dir)
    model_file = data_dir / "artista/models/cogent/model.pt"
    dataset_file = f"{data_dir}/{dataset}"
    s2_1 = data_dir / "artista/preprocessed/2DPI_1.h5ad"
    s2_2 = data_dir / "artista/preprocessed/2DPI_2.h5ad"
    s2_3 = data_dir / "artista/preprocessed/2DPI_3.h5ad"
    s5_1 = data_dir / "artista/preprocessed/5DPI_1.h5ad"
    s5_2 = data_dir / "artista/preprocessed/5DPI_2.h5ad"
    s5_3 = data_dir / "artista/preprocessed/5DPI_3.h5ad"
    s10_1 = data_dir / "artista/preprocessed/10DPI_1.h5ad"
    s10_2 = data_dir / "artista/preprocessed/10DPI_2.h5ad"
    s10_3 = data_dir / "artista/preprocessed/10DPI_3.h5ad"
    s15_1 = data_dir / "artista/preprocessed/15DPI_1.h5ad"
    s15_2 = data_dir / "artista/preprocessed/15DPI_2.h5ad"
    s15_3 = data_dir / "artista/preprocessed/15DPI_3.h5ad"
    s15_4 = data_dir / "artista/preprocessed/15DPI_4.h5ad"
    SAMPLE_FILES = [
        (s2_1, 2),
        (s2_2, 2),
        (s2_3, 2),
        (s5_1, 5),
        (s5_2, 5),
        (s5_3, 5),
        (s10_1, 10),
        (s10_2, 10),
        (s10_3, 10),
        (s15_1, 15),
        (s15_2, 15),
        (s15_3, 15),
        (s15_4, 15),
    ]
    SPATIAL_CLUSTERS = [2, 4, 6]

    def load_model():
        return Cogent.load(Path(model_file).parent, device="cpu")

    def load_adata():
        return ad.read_h5ad(dataset_file)

    def spatial_inlier_mask(xy):
        keep = np.isfinite(xy).all(axis=1)
        for axis in range(2):
            vals = xy[keep, axis]
            if vals.size < 10:
                continue
            (lo, hi) = np.quantile(vals, [0.005, 0.995])
            keep &= (xy[:, axis] >= lo) & (xy[:, axis] <= hi)
        return keep

    def build_cluster_means(model, a, mask):
        return {
            int(k): np.asarray(v, dtype=np.float32)[mask]
            for (k, v) in cluster_means_all(
                model, a, view="average", resolution=resolution
            ).items()
        }

    def build_injury_mask(a, mask):
        zones = (
            a.obs["inj_zone"].astype(str).to_numpy()
            if "inj_zone" in a.obs
            else np.array(["unknown"] * a.n_obs)
        )
        values = np.isin(zones, ["wound", "inj_nonwound"]).astype(np.float32)[mask]
        return {"values": values, "rule": "inj_zone in wound/inj_nonwound"}

    def build_proximal_distal(sample_scores):
        keep = sample_scores[sample_scores["cluster"].isin(SPATIAL_CLUSTERS)].copy()
        grouped = (
            keep.groupby(["cluster", "dpi", "inj_uninj"])["mean_score"]
            .apply(list)
            .reset_index()
        )
        return grouped

    def build_spread_distance(model):
        rows_spread = []
        rows_dist = []
        for sample_path, dpi in SAMPLE_FILES:
            a = ad.read_h5ad(sample_path)
            raw = np.asarray(a.obsm["spatial"], dtype=np.float32)
            mask = spatial_inlier_mask(raw)
            xy = raw[mask]
            cm = build_cluster_means(model, a, mask)
            injury = build_injury_mask(a, mask)["values"] > 0
            if injury.sum() > 0 and (~injury).sum() > 0:
                tree = cKDTree(xy[injury])
                dist = tree.query(xy, k=1)[0]
            else:
                dist = np.zeros(xy.shape[0], dtype=np.float32)
            edges = np.quantile(dist, np.linspace(0, 1, 6))
            edges = np.unique(edges)
            for cluster in SPATIAL_CLUSTERS:
                vals = np.asarray(cm[int(cluster)], dtype=np.float32)
                thr = float(np.quantile(vals, 0.8))
                rows_spread.append(
                    {
                        "dpi": int(dpi),
                        "cluster": int(cluster),
                        "high_fraction": float((vals >= thr).mean()),
                    }
                )
                for i in range(len(edges) - 1):
                    keep = (dist >= edges[i]) & (dist <= edges[i + 1])
                    if keep.sum() == 0:
                        continue
                    rows_dist.append(
                        {
                            "dpi": int(dpi),
                            "cluster": int(cluster),
                            "bin": int(i),
                            "distance": float((edges[i] + edges[i + 1]) / 2),
                            "mean_score": float(np.mean(vals[keep])),
                        }
                    )
        return {
            "spread": pd.DataFrame(rows_spread),
            "distance": pd.DataFrame(rows_dist),
        }

    model = load_model()
    adata = load_adata()
    raw_xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    inlier_mask = spatial_inlier_mask(raw_xy)
    spatial_xy = raw_xy[inlier_mask]
    cluster_means_by_cluster = build_cluster_means(model, adata, inlier_mask)
    sample_scores = compute_cluster_sample_scores(
        model, SAMPLE_FILES, resolution=resolution
    )
    injury_mask_payload = build_injury_mask(adata, inlier_mask)
    proximal_distal_df = build_proximal_distal(sample_scores)
    spread_distance_payload = build_spread_distance(model)
    spatial_spread_df = spread_distance_payload["spread"]
    return {
        "spatial_xy": spatial_xy,
        "cluster_means_by_cluster": cluster_means_by_cluster,
        "sample_scores": sample_scores,
        "injury_mask_payload": injury_mask_payload,
        "proximal_distal_df": proximal_distal_df,
        "spatial_spread_df": spatial_spread_df,
    }


def panel_s12_timeline(sample_scores):
    series = []
    for cluster, sub in sample_scores[sample_scores["cluster"].isin([2, 4, 6])].groupby(
        "cluster", sort=False
    ):
        pooled = sub.groupby("dpi")["mean_score"].mean().reset_index()
        points = [
            {"x": float(r["dpi"]), "y": float(r["mean_score"])}
            for (_, r) in pooled.sort_values("dpi").iterrows()
        ]
        series.append({"name": f"c{int(cluster)}", "points": points})
    return {"series": series}


def panel_s12_mask(spatial_xy, injury_mask_payload):
    return {
        "x": spatial_xy[:, 0],
        "y": -spatial_xy[:, 1],
        "values": injury_mask_payload["values"],
        "type": "continuous",
        "_meta": {"rule": injury_mask_payload["rule"]},
    }


def panel_s12_prox_dist(proximal_distal_df):
    groups = []
    for (cluster, zone), sub in proximal_distal_df.groupby(
        ["cluster", "inj_uninj"], sort=False
    ):
        vals = []
        for item in sub["mean_score"]:
            vals.extend([float(x) for x in item])
        groups.append({"label": f"c{int(cluster)}\n{zone}", "values": vals})
    return {"groups": groups}


def panel_s12_spread(spatial_spread_df):
    series = []
    for cluster, sub in spatial_spread_df.groupby("cluster", sort=False):
        points = [
            {"x": float(r["dpi"]), "y": float(r["high_fraction"])}
            for (_, r) in sub.groupby("dpi")["high_fraction"]
            .mean()
            .reset_index()
            .sort_values("dpi")
            .iterrows()
        ]
        series.append({"name": f"c{int(cluster)}", "points": points})
    return {"series": series}


def calculate_maps(
    data_dir, dataset="artista/preprocessed/2DPI_1.h5ad", resolution=1.0
):
    data_dir = Path(data_dir)
    model_file = data_dir / "artista/models/cogent/model.pt"
    dataset_file = f"{data_dir}/{dataset}"

    def load_model():
        return Cogent.load(Path(model_file).parent, device="cpu")

    def load_adata():
        return ad.read_h5ad(dataset_file)

    def spatial_inlier_mask(xy):
        keep = np.isfinite(xy).all(axis=1)
        for axis in range(2):
            vals = xy[keep, axis]
            if vals.size < 10:
                continue
            (lo, hi) = np.quantile(vals, [0.005, 0.995])
            keep &= (xy[:, axis] >= lo) & (xy[:, axis] <= hi)
        return keep

    def build_cluster_means(model, a, mask):
        return {
            int(k): np.asarray(v, dtype=np.float32)[mask]
            for (k, v) in cluster_means_all(
                model, a, view="average", resolution=resolution
            ).items()
        }

    model = load_model()
    adata = load_adata()
    raw_xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    inlier_mask = spatial_inlier_mask(raw_xy)
    spatial_xy = raw_xy[inlier_mask]
    cluster_means_by_cluster = build_cluster_means(model, adata, inlier_mask)
    return {
        "cluster_means_by_cluster": cluster_means_by_cluster,
        "spatial_xy": spatial_xy,
    }


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(
        data_dir, dataset="artista/preprocessed/2DPI_1.h5ad", resolution=1.0
    )
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s12_timeline(sample_scores=results["sample_scores"]),
        "line",
        output_dir / "s12-timeline",
        "Regeneration timeline scores",
        {"xlabel": "DPI", "ylabel": "mean score"},
    )
    save_panel(
        panel_s12_mask(
            spatial_xy=results["spatial_xy"],
            injury_mask_payload=results["injury_mask_payload"],
        ),
        "spatial",
        output_dir / "s12-mask",
        "Injury-region mask",
        {},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"], results["cluster_means_by_cluster"][int(2)]
        ),
        "spatial",
        output_dir / "s12-wound-2dpi",
        "Wound/radial-glia 2 DPI",
        {},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"], results["cluster_means_by_cluster"][int(4)]
        ),
        "spatial",
        output_dir / "s12-microglia-2dpi",
        "Macrophage/microglia 2 DPI",
        {},
    )
    save_panel(
        panel_s12_prox_dist(proximal_distal_df=results["proximal_distal_df"]),
        "boxplot",
        output_dir / "s12-prox-dist",
        "Proximal versus distal scores",
        {"ylabel": "mean score"},
    )
    save_panel(
        panel_s12_spread(spatial_spread_df=results["spatial_spread_df"]),
        "line",
        output_dir / "s12-spread",
        "Spatial spread over time",
        {"xlabel": "DPI", "ylabel": "high-score fraction"},
    )
    del results
    results = calculate_maps(
        data_dir, dataset="artista/preprocessed/10DPI_1.h5ad", resolution=1.0
    )
    with (output_dir / "analysis_2.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        spatial_panel(
            results["spatial_xy"], results["cluster_means_by_cluster"][int(2)]
        ),
        "spatial",
        output_dir / "s12-wound-10dpi",
        "Wound/radial-glia 10 DPI",
        {},
    )
    del results
    results = calculate_maps(
        data_dir, dataset="artista/preprocessed/15DPI_1.h5ad", resolution=1.0
    )
    with (output_dir / "analysis_3.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        spatial_panel(
            results["spatial_xy"], results["cluster_means_by_cluster"][int(6)]
        ),
        "spatial",
        output_dir / "s12-stress-15dpi",
        "Stress/remodeling 15 DPI",
        {},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s11_axolotl_spatial_transition")
