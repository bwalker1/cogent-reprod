from pathlib import Path
from cogent import Cogent
from cogent.utils import cluster_means_all
from cogent.analysis import compute_cluster_sample_scores
import anndata as ad
import numpy as np
import pandas as pd
from umap import UMAP
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(data_dir, dataset="artista/preprocessed/2DPI_1.h5ad"):
    data_dir = Path(data_dir)
    model_path = "artista/models/cogent"
    preproc_dir = data_dir / "artista/preprocessed"
    model_file = f"{data_dir}/{model_path}/model.pt"
    cluster_labels_file = f"{data_dir}/{model_path}/cluster_labels_res1.0.csv"
    dataset_file = f"{data_dir}/{dataset}"
    s2_1 = f"{preproc_dir}/2DPI_1.h5ad"
    s2_2 = f"{preproc_dir}/2DPI_2.h5ad"
    s2_3 = f"{preproc_dir}/2DPI_3.h5ad"
    s5_1 = f"{preproc_dir}/5DPI_1.h5ad"
    s5_2 = f"{preproc_dir}/5DPI_2.h5ad"
    s5_3 = f"{preproc_dir}/5DPI_3.h5ad"
    s10_1 = f"{preproc_dir}/10DPI_1.h5ad"
    s10_2 = f"{preproc_dir}/10DPI_2.h5ad"
    s10_3 = f"{preproc_dir}/10DPI_3.h5ad"
    s15_1 = f"{preproc_dir}/15DPI_1.h5ad"
    s15_2 = f"{preproc_dir}/15DPI_2.h5ad"
    s15_3 = f"{preproc_dir}/15DPI_3.h5ad"
    s15_4 = f"{preproc_dir}/15DPI_4.h5ad"
    ARTISTA_SAMPLES_WITH_PATHS = [
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
    DPI_ORDER = [2, 5, 10, 15]
    CLUSTER_RESOLUTION = 1.0
    model = Cogent.load(Path(model_file).parent, device="cpu")
    adata = ad.read_h5ad(dataset_file)

    def _build_cluster_labels(m, resolution, labels_path):
        labs = np.asarray(m._leiden_results[resolution]["labels"]["average"], dtype=int)
        (clusters, counts) = np.unique(labs, return_counts=True)
        annotations = pd.read_csv(labels_path)
        annotations["cluster"] = annotations["cluster"].astype(int)
        if annotations["cluster"].duplicated().any():
            dupes = sorted(
                annotations.loc[annotations["cluster"].duplicated(), "cluster"].unique()
            )
            raise ValueError(f"duplicate cluster annotations: {dupes}")
        label_map = dict(zip(annotations["cluster"], annotations["label"].astype(str)))
        missing = sorted(set(clusters.astype(int)) - set(label_map))
        if missing:
            raise ValueError(f"missing cluster annotations: {missing}")
        return pd.DataFrame(
            {
                "cluster": clusters.astype(int),
                "label": [label_map[int(c)] for c in clusters],
                "n_genes": counts.astype(int),
            }
        )

    labels_df = _build_cluster_labels(model, CLUSTER_RESOLUTION, cluster_labels_file)

    def _build_shared_umap_payload(m, labels, resolution):
        coords = np.asarray(m._umap_coords["average"], dtype=np.float32)
        labs = np.asarray(m._leiden_results[resolution]["labels"]["average"], dtype=int)
        label_map = dict(
            zip(labels["cluster"].astype(int), labels["label"].astype(str))
        )
        size_map = dict(
            zip(labels["cluster"].astype(int), labels["n_genes"].astype(int))
        )

        def fmt(c):
            return (
                f"{int(c)}: {label_map[int(c)]}" if int(c) in label_map else str(int(c))
            )

        threshold = 20
        keep = np.asarray(
            [int(size_map.get(int(c), 0)) >= threshold for c in labs], dtype=bool
        )
        return {
            "x": np.asarray(coords[keep, 0], dtype=np.float32),
            "y": np.asarray(coords[keep, 1], dtype=np.float32),
            "values": [fmt(c) for c in labs[keep]],
            "type": "categorical",
            "_meta": {
                "cluster_sizes": {
                    fmt(int(r["cluster"])): int(r["n_genes"])
                    for (_, r) in labels.iterrows()
                }
            },
        }

    shared_umap_payload = _build_shared_umap_payload(
        model, labels_df, CLUSTER_RESOLUTION
    )

    def _spatial_inlier_mask(xy):
        keep = np.isfinite(xy).all(axis=1)
        for axis in range(2):
            while True:
                vals = xy[keep, axis]
                if vals.size < 10:
                    break
                sorted_vals = np.sort(vals)
                (q005, q995) = np.quantile(vals, [0.005, 0.995])
                robust_span = max(float(q995 - q005), 1.0)
                diffs = np.diff(sorted_vals)
                tail_start = max(0, int(0.95 * len(sorted_vals)) - 1)
                tail_diffs = diffs[tail_start:]
                if tail_diffs.size == 0:
                    break
                rel_idx = int(np.argmax(tail_diffs))
                gap = float(tail_diffs[rel_idx])
                if gap <= 0.2 * robust_span:
                    break
                cutoff = float(sorted_vals[tail_start + rel_idx])
                keep &= xy[:, axis] <= cutoff
        return keep

    raw_spatial_xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    spatial_inlier_mask = _spatial_inlier_mask(raw_spatial_xy)
    spatial_xy = raw_spatial_xy[spatial_inlier_mask]

    def _cluster_means_by_cluster(m, a, mask):
        return {
            int(k): np.asarray(v, dtype=np.float32)[mask]
            for (k, v) in cluster_means_all(
                m, a, view="average", resolution=CLUSTER_RESOLUTION
            ).items()
        }

    cluster_means_by_cluster = _cluster_means_by_cluster(
        model, adata, spatial_inlier_mask
    )

    def _build_shared_umap_by_stage(m):
        embeddings = np.asarray(m.get_embeddings(), dtype=np.float32)
        half_dim = embeddings.shape[-1] // 2
        first_half = embeddings[:, :, :half_dim]
        (n_stages, n_genes, _) = first_half.shape
        coords = (
            UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="cosine",
                init="spectral",
                random_state=42,
            )
            .fit_transform(first_half.reshape(-1, half_dim))
            .astype(np.float32)
        )
        group_ids = [str(g) for g in m.group_ids or []]
        stage_labels = [
            group_ids[s] if s < len(group_ids) else f"stage {s + 1}"
            for s in range(n_stages)
        ]
        return {
            "x": np.asarray(coords[:, 0], dtype=np.float32),
            "y": np.asarray(coords[:, 1], dtype=np.float32),
            "values": [
                stage_labels[s] for s in range(n_stages) for _ in range(n_genes)
            ],
            "type": "categorical",
        }

    shared_umap_by_stage = _build_shared_umap_by_stage(model)
    sample_scores = compute_cluster_sample_scores(
        model, ARTISTA_SAMPLES_WITH_PATHS, resolution=CLUSTER_RESOLUTION
    )

    def _build_inj_uninj_wide(scores):
        pooled = (
            scores.groupby(["cluster", "dpi", "inj_uninj"])["mean_score"]
            .mean()
            .unstack("inj_uninj")
            .reset_index()
        )
        pooled["log2"] = np.log2(pooled["inj"] / pooled["uninj"])
        wide = pooled.pivot(index="cluster", columns="dpi", values="log2").reset_index()
        return wide.rename(columns={d: f"{d}DPI" for d in DPI_ORDER})

    inj_uninj_mat = (
        labels_df.merge(_build_inj_uninj_wide(sample_scores), on="cluster")
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    def _build_cluster_shift_mat(m, labels, resolution, dpi_order):
        E = m._embeddings
        half = E.shape[-1] // 2
        E_mag = E[..., :half]
        gid = list(m.group_ids or [])
        idxs = [gid.index(f"{d}DPI") for d in dpi_order]
        E_ord = E_mag[idxs]
        lab = np.asarray(m._leiden_results[resolution]["labels"]["average"], dtype=int)
        trans = [(dpi_order[i], dpi_order[i + 1]) for i in range(len(dpi_order) - 1)]
        cols = [f"{a}→{b}" for (a, b) in trans]
        rows = []
        for _, r in labels.iterrows():
            c = int(r["cluster"])
            mask = lab == c
            n = int(mask.sum())
            if n == 0:
                continue
            cent = E_ord[:, mask, :].mean(axis=1)
            norms = np.linalg.norm(cent, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            u = cent / norms[:, None]
            dists = [
                float(1.0 - np.dot(u[i], u[i + 1])) for i in range(len(dpi_order) - 1)
            ]
            rec = {"cluster": int(c), "label": r["label"], "n_genes": n}
            for cn, d in zip(cols, dists):
                rec[cn] = d
            rec["mean_shift"] = float(np.mean(dists))
            rows.append(rec)
        return pd.DataFrame(rows)

    cluster_shift_mat = _build_cluster_shift_mat(
        model, labels_df, CLUSTER_RESOLUTION, DPI_ORDER
    )
    return {
        "shared_umap_payload": shared_umap_payload,
        "spatial_xy": spatial_xy,
        "cluster_means_by_cluster": cluster_means_by_cluster,
        "shared_umap_by_stage": shared_umap_by_stage,
        "inj_uninj_mat": inj_uninj_mat,
        "cluster_shift_mat": cluster_shift_mat,
    }


def panel_heatmap_inj_uninj(inj_uninj_mat):
    threshold = 20
    df = inj_uninj_mat
    dpis_ordered = [2, 5, 10, 15]
    columns = [f"{d}DPI" for d in dpis_ordered]
    rows = []
    values = []
    row_sizes = []
    for _, r in df.iterrows():
        if int(r["n_genes"]) < threshold:
            continue
        rows.append(f"{int(r['cluster'])}: {r['label']}")
        values.append([float(r[f"{d}DPI"]) for d in dpis_ordered])
        row_sizes.append(int(r["n_genes"]))
    symmetric = True
    return {"rows": rows, "columns": columns, "values": values, "symmetric": symmetric}


def panel_line_trajectory(inj_uninj_mat):
    keep_labels = {"Lymphoid/immune", "Macrophage/microglia", "Stress+remodeling"}
    threshold = 20
    df = inj_uninj_mat
    dpis_ordered = [2, 5, 10, 15]
    series = []
    for _, r in df.iterrows():
        lab = r["label"]
        if lab not in keep_labels:
            continue
        if int(r["n_genes"]) < threshold:
            continue
        points = [{"x": d, "y": float(r[f"{d}DPI"])} for d in dpis_ordered]
        series.append({"name": lab, "points": points})
    return {"series": series}


def panel_umap_by_stage(shared_umap_by_stage):
    return {
        "x": -shared_umap_by_stage["x"],
        "y": -shared_umap_by_stage["y"],
        "values": shared_umap_by_stage["values"],
        "type": shared_umap_by_stage["type"],
    }


def panel_heatmap_dpi_shift(cluster_shift_mat):
    threshold = 20
    df = cluster_shift_mat.sort_values("mean_shift", ascending=False)
    transition_cols = [c for c in df.columns if "→" in c]
    columns = []
    mat = []
    for _, r in df.iterrows():
        if int(r["n_genes"]) < threshold:
            continue
        columns.append(str(int(r["cluster"])))
        mat.append([float(r[c]) for c in transition_cols])
    rows = [c.replace("→", " → ") for c in transition_cols]
    values = [[mat[j][i] for j in range(len(mat))] for i in range(len(transition_cols))]
    return {"rows": rows, "columns": columns, "values": values}


def panel_aggregate_dpi_shift(cluster_shift_mat):
    threshold = 20
    df = cluster_shift_mat[cluster_shift_mat["n_genes"] >= threshold].copy()
    transition_cols = [c for c in df.columns if "→" in c]
    points = []
    for col in transition_cols:
        later = float(col.split("→")[1].replace("DPI", ""))
        weights = df["n_genes"].astype(float)
        mean_shift = float((df[col] * weights).sum() / weights.sum())
        points.append({"x": later, "y": mean_shift})
    series = [{"name": "weighted module mean", "points": points, "color": "#d8915f"}]
    return {"series": series}


def panel_module_response_delta(inj_uninj_mat):
    import pandas as pd

    threshold = 20
    df = inj_uninj_mat[inj_uninj_mat["n_genes"] >= threshold].copy()
    df["delta_15_vs_2"] = df["15DPI"] - df["2DPI"]
    hi = df.sort_values("delta_15_vs_2", ascending=False).head(6)
    lo = df.sort_values("delta_15_vs_2", ascending=True).head(6)
    df = pd.concat([hi, lo], ignore_index=True).drop_duplicates("cluster")
    df = df.sort_values("delta_15_vs_2", ascending=False)
    items = [
        {"label": str(int(r["cluster"])), "value": float(r["delta_15_vs_2"])}
        for (_, r) in df.iterrows()
    ]
    bars = [{"name": "modules", "items": items}]
    return {"bars": bars}


def calculate_maps(data_dir, dataset="artista/preprocessed/2DPI_1.h5ad"):
    data_dir = Path(data_dir)
    model_path = "artista/models/cogent"
    model_file = f"{data_dir}/{model_path}/model.pt"
    dataset_file = f"{data_dir}/{dataset}"
    CLUSTER_RESOLUTION = 1.0
    model = Cogent.load(Path(model_file).parent, device="cpu")
    adata = ad.read_h5ad(dataset_file)

    def _spatial_inlier_mask(xy):
        keep = np.isfinite(xy).all(axis=1)
        for axis in range(2):
            while True:
                vals = xy[keep, axis]
                if vals.size < 10:
                    break
                sorted_vals = np.sort(vals)
                (q005, q995) = np.quantile(vals, [0.005, 0.995])
                robust_span = max(float(q995 - q005), 1.0)
                diffs = np.diff(sorted_vals)
                tail_start = max(0, int(0.95 * len(sorted_vals)) - 1)
                tail_diffs = diffs[tail_start:]
                if tail_diffs.size == 0:
                    break
                rel_idx = int(np.argmax(tail_diffs))
                gap = float(tail_diffs[rel_idx])
                if gap <= 0.2 * robust_span:
                    break
                cutoff = float(sorted_vals[tail_start + rel_idx])
                keep &= xy[:, axis] <= cutoff
        return keep

    raw_spatial_xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    spatial_inlier_mask = _spatial_inlier_mask(raw_spatial_xy)
    spatial_xy = raw_spatial_xy[spatial_inlier_mask]

    def _cluster_means_by_cluster(m, a, mask):
        return {
            int(k): np.asarray(v, dtype=np.float32)[mask]
            for (k, v) in cluster_means_all(
                m, a, view="average", resolution=CLUSTER_RESOLUTION
            ).items()
        }

    cluster_means_by_cluster = _cluster_means_by_cluster(
        model, adata, spatial_inlier_mask
    )
    return {
        "cluster_means_by_cluster": cluster_means_by_cluster,
        "spatial_xy": spatial_xy,
    }


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, dataset="artista/preprocessed/2DPI_1.h5ad")
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        results["shared_umap_payload"],
        "umap",
        output_dir / "umap-avg",
        "UMAP",
        {},
    )
    save_panel(
        panel_heatmap_inj_uninj(inj_uninj_mat=results["inj_uninj_mat"]),
        "heatmap",
        output_dir / "heatmap-inj-uninj",
        "log₂(inj / uninj) across DPI",
        {"xlabel": "", "ylabel": ""},
    )
    save_panel(
        panel_line_trajectory(inj_uninj_mat=results["inj_uninj_mat"]),
        "line",
        output_dir / "line-trajectory",
        "Module trajectory by DPI",
        {"xlabel": "Days post-injury", "ylabel": "log₂(inj / uninj)"},
    )
    save_panel(
        panel_umap_by_stage(shared_umap_by_stage=results["shared_umap_by_stage"]),
        "umap",
        output_dir / "umap-by-stage",
        "UMAP by stage",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means_by_cluster"][9]),
        "spatial",
        output_dir / "spatial-lymphoid-2dpi",
        "2 DPI",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means_by_cluster"][4]),
        "spatial",
        output_dir / "spatial-macrophage-2dpi",
        "2 DPI",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means_by_cluster"][6]),
        "spatial",
        output_dir / "spatial-stress-2dpi",
        "2 DPI",
        {},
    )
    save_panel(
        panel_heatmap_dpi_shift(cluster_shift_mat=results["cluster_shift_mat"]),
        "heatmap",
        output_dir / "heatmap-dpi-shift",
        "Module embedding shift",
        {"xlabel": "cluster", "ylabel": ""},
    )
    save_panel(
        panel_aggregate_dpi_shift(cluster_shift_mat=results["cluster_shift_mat"]),
        "line",
        output_dir / "aggregate-dpi-shift",
        "Aggregate module shift",
        {"xlabel": "Days post-injury", "ylabel": "Mean module shift", "ymin": 0},
    )
    save_panel(
        panel_module_response_delta(inj_uninj_mat=results["inj_uninj_mat"]),
        "divergent",
        output_dir / "module-response-delta",
        "15 vs 2 DPI response",
        {"ylabel": "Δ log₂(inj / uninj)"},
    )
    del results
    for index, dpi in enumerate((5, 10), start=2):
        results = calculate_maps(
            data_dir, dataset=f"artista/preprocessed/{dpi}DPI_1.h5ad"
        )
        with (output_dir / f"analysis_{index}.pkl").open("wb") as handle:
            pickle.dump(results, handle, protocol=5)
        for name, cluster in (("lymphoid", 9), ("macrophage", 4), ("stress", 6)):
            save_panel(
                spatial_panel(
                    results["spatial_xy"], results["cluster_means_by_cluster"][cluster]
                ),
                "spatial",
                output_dir / f"spatial-{name}-{dpi}dpi",
                f"{dpi} DPI",
                {},
            )
        del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "fig6")
