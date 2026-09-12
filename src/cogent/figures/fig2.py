from pathlib import Path
from cogent import Cogent
from cogent.utils import cluster_means_all
from cogent.analysis import compute_cluster_stage_shifts, compute_gene_similarity_matrix
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from umap import UMAP
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(
    data_dir, view="average", resolution=0.5, dataset="E12.5_E1S1", gene="Mt2"
):
    data_dir = Path(data_dir)
    model_path = "mosta/models/cogent"
    model_file = f"{data_dir}/{model_path}/model.pt"
    labels_file = f"{data_dir}/{model_path}/cluster_labels_res0.5.csv"
    dataset_file = f"{data_dir}/mosta/raw/{dataset}.MOSTA.h5ad"
    e125_file = data_dir / "mosta/raw/E12.5_E1S1.MOSTA.h5ad"
    e135_file = data_dir / "mosta/raw/E13.5_E1S1.MOSTA.h5ad"

    loaded_model = Cogent.load(Path(model_file).parent, device="cpu")

    loaded_adata = ad.read_h5ad(dataset_file)

    spatial_xy = np.asarray(loaded_adata.obsm["spatial"], dtype=np.float32)

    def build_shared_umap():
        model = loaded_model
        coords = model._umap_coords["average"]
        labs = [int(label) for label in model._leiden_results[0.5]["labels"]["average"]]
        labels_df = pd.read_csv(labels_file)
        label_map = dict(zip(labels_df["cluster"], labels_df["label"]))

        def fmt(c):
            return f"{c}: {label_map[c]}" if c in label_map else str(c)

        return {
            "x": np.asarray(coords[:, 0], dtype=np.float32),
            "y": np.asarray(coords[:, 1], dtype=np.float32),
            "values": [fmt(label) for label in labs],
            "type": "categorical",
            "_meta": {
                "cluster_sizes": {
                    fmt(int(r["cluster"])): int(r["n_genes"])
                    for (_, r) in labels_df.iterrows()
                }
            },
        }

    shared_umap = build_shared_umap()

    def build_shared_umap_by_stage():
        model = loaded_model
        embeddings = model.get_embeddings()
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
        STAGE_LABELS = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"][:n_stages]
        return {
            "x": np.asarray(coords[:, 0], dtype=np.float32),
            "y": np.asarray(coords[:, 1], dtype=np.float32),
            "values": [
                STAGE_LABELS[s] for s in range(n_stages) for _ in range(n_genes)
            ],
            "type": "categorical",
        }

    shared_umap_by_stage = build_shared_umap_by_stage()

    def build_cluster_means():
        cms = cluster_means_all(
            loaded_model, loaded_adata, view=view, resolution=resolution
        )
        if isinstance(cms, dict):
            return {k: np.asarray(v, dtype=np.float32) for (k, v) in cms.items()}
        return np.asarray(cms, dtype=np.float32)

    cluster_means = build_cluster_means()
    gene_expr = np.asarray(
        loaded_adata[:, gene].X.toarray().ravel()
        if sp.issparse(loaded_adata[:, gene].X)
        else np.asarray(loaded_adata[:, gene].X).ravel(),
        dtype=np.float32,
    )
    cluster_shift_mat = compute_cluster_stage_shifts(
        loaded_model, resolution=resolution
    ).merge(
        pd.read_csv(labels_file)[["cluster", "label"]],
        left_index=True,
        right_on="cluster",
    )

    def build_gene_sim():
        out = compute_gene_similarity_matrix(loaded_model, resolution=resolution)
        out["values"] = np.asarray(out["values"], dtype=np.float32)
        return out

    def build_e125_e135_cluster_log2fc():
        model = loaded_model
        labels_df = pd.read_csv(labels_file)
        a125 = ad.read_h5ad(e125_file)
        a135 = ad.read_h5ad(e135_file)
        means125 = cluster_means_all(model, a125, view=view, resolution=resolution)
        means135 = cluster_means_all(model, a135, view=view, resolution=resolution)
        rows = []
        for _, r in labels_df.iterrows():
            cid = int(r["cluster"])
            if cid not in means125 or cid not in means135:
                continue
            v125 = float(np.nanmean(np.asarray(means125[cid], dtype=np.float32)))
            v135 = float(np.nanmean(np.asarray(means135[cid], dtype=np.float32)))
            rows.append(
                {
                    "cluster": cid,
                    "label": r["label"],
                    "n_genes": int(r["n_genes"]),
                    "log2_e135_over_e125": float(
                        np.log2((v135 + 1e-06) / (v125 + 1e-06))
                    ),
                }
            )
        return pd.DataFrame(rows)

    e125_e135_cluster_log2fc = build_e125_e135_cluster_log2fc()

    def build_e125_e135_gene_shift():
        model = loaded_model
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        emb_mag = emb[:, :, :half]
        group_ids = [str(g) for g in model.group_ids or []]
        start_idx = group_ids.index("12") if "12" in group_ids else 0
        end_idx = group_ids.index("13") if "13" in group_ids else 1
        a = emb_mag[start_idx]
        b = emb_mag[end_idx]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-09)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-09)
        shift = 1.0 - np.sum(a_norm * b_norm, axis=1)
        return (
            pd.DataFrame(
                {
                    "gene": np.asarray(model._gene_names, dtype=str),
                    "embedding_shift": np.asarray(shift, dtype=np.float32),
                    "from_stage": "E12.5",
                    "to_stage": "E13.5",
                }
            )
            .sort_values("embedding_shift", ascending=False)
            .reset_index(drop=True)
        )

    e125_e135_gene_shift = build_e125_e135_gene_shift()
    gene_sim = build_gene_sim()
    return {
        "spatial_xy": spatial_xy,
        "shared_umap": shared_umap,
        "shared_umap_by_stage": shared_umap_by_stage,
        "cluster_means": cluster_means,
        "gene_expr": gene_expr,
        "cluster_shift_mat": cluster_shift_mat,
        "e125_e135_cluster_log2fc": e125_e135_cluster_log2fc,
        "e125_e135_gene_shift": e125_e135_gene_shift,
        "gene_sim": gene_sim,
    }


def panel_shared_umap_by_stage(shared_umap_by_stage):
    return {
        "x": -shared_umap_by_stage["x"],
        "y": -shared_umap_by_stage["y"],
        "values": shared_umap_by_stage["values"],
        "type": shared_umap_by_stage["type"],
    }


def panel_heatmap_cluster_shift(cluster_shift_mat):
    threshold = 20
    df = cluster_shift_mat.sort_values("mean_shift", ascending=False)
    transition_cols = [c for c in df.columns if "→" in c]
    columns = []
    mat = []
    for _, r in df.iterrows():
        if int(r["n_genes"]) < threshold:
            continue
        columns.append(r["label"])
        mat.append([float(r[c]) for c in transition_cols])
    rows = [c.replace("→", " → ") for c in transition_cols]
    values = [[mat[j][i] for j in range(len(mat))] for i in range(len(transition_cols))]
    return {"rows": rows, "columns": columns, "values": values}


def panel_global_stage_change(cluster_shift_mat):
    threshold = 20
    df = cluster_shift_mat[cluster_shift_mat["n_genes"] >= threshold].copy()
    transition_cols = [c for c in df.columns if "→" in c]
    points = []
    for col in transition_cols:
        later = float(col.split("→")[1]) + 0.5
        weights = df["n_genes"].astype(float)
        mean_shift = float((df[col] * weights).sum() / weights.sum())
        points.append({"x": later, "y": mean_shift})
    series = [{"name": "weighted cluster mean", "points": points, "color": "#d8915f"}]
    return {"series": series}


def panel_e125_e135_module_log2fc(e125_e135_cluster_log2fc):
    threshold = 20
    df = e125_e135_cluster_log2fc[
        e125_e135_cluster_log2fc["n_genes"] >= threshold
    ].copy()
    df = df.sort_values("log2_e135_over_e125", ascending=False).head(12)
    items = [
        {"label": str(int(r["cluster"])), "value": float(r["log2_e135_over_e125"])}
        for (_, r) in df.iterrows()
    ]
    bars = [{"name": "modules", "items": items}]
    return {"bars": bars}


def panel_clustermap_gene_sim(gene_sim):
    n = len(gene_sim["genes"])
    rows = [""] * n
    columns = rows
    values = gene_sim["values"].tolist()
    rowGroups = [str(c) for c in gene_sim["clusters"]]
    symmetric = True
    return {
        "rows": rows,
        "columns": columns,
        "values": values,
        "rowGroups": rowGroups,
        "symmetric": symmetric,
    }


def panel_e125_e135_gene_shift(e125_e135_gene_shift):
    n = 6
    df = e125_e135_gene_shift.sort_values("embedding_shift", ascending=False).head(n)
    bars = [
        {"label": str(r["gene"]), "value": float(r["embedding_shift"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def calculate_maps(
    data_dir, view="average", resolution=0.5, dataset="E12.5_E1S1", gene="Mt2"
):
    data_dir = Path(data_dir)
    model_path = "mosta/models/cogent"
    model_file = f"{data_dir}/{model_path}/model.pt"
    dataset_file = f"{data_dir}/mosta/raw/{dataset}.MOSTA.h5ad"

    loaded_model = Cogent.load(Path(model_file).parent, device="cpu")

    loaded_adata = ad.read_h5ad(dataset_file)

    spatial_xy = np.asarray(loaded_adata.obsm["spatial"], dtype=np.float32)

    def build_cluster_means():
        cms = cluster_means_all(
            loaded_model, loaded_adata, view=view, resolution=resolution
        )
        if isinstance(cms, dict):
            return {k: np.asarray(v, dtype=np.float32) for (k, v) in cms.items()}
        return np.asarray(cms, dtype=np.float32)

    cluster_means = build_cluster_means()
    gene_expr = np.asarray(
        loaded_adata[:, gene].X.toarray().ravel()
        if sp.issparse(loaded_adata[:, gene].X)
        else np.asarray(loaded_adata[:, gene].X).ravel(),
        dtype=np.float32,
    )
    return {
        "cluster_means": cluster_means,
        "gene_expr": gene_expr,
        "spatial_xy": spatial_xy,
    }


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(
        data_dir, view="average", resolution=0.5, dataset="E12.5_E1S1", gene="Mt2"
    )
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        results["shared_umap"],
        "umap",
        output_dir / "shared-umap",
        "Shared UMAP",
        {},
    )
    save_panel(
        panel_shared_umap_by_stage(
            shared_umap_by_stage=results["shared_umap_by_stage"]
        ),
        "umap",
        output_dir / "shared-umap-by-stage",
        "UMAP by stage",
        {},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][0],
            results["cluster_means"][1],
        ),
        "spatial",
        output_dir / "spatial-cluster",
        "E12.5",
        {"channel_labels": ["Module 0", "Module 1"]},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][7],
            results["cluster_means"][11],
        ),
        "spatial",
        output_dir / "spatial-cm",
        "E12.5",
        {"channel_labels": ["Module 7", "Module 11"]},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["gene_expr"]),
        "spatial",
        output_dir / "spatial-mt2-e125",
        "Mt2 E12.5",
        {},
    )
    save_panel(
        panel_heatmap_cluster_shift(cluster_shift_mat=results["cluster_shift_mat"]),
        "heatmap",
        output_dir / "heatmap-cluster-shift",
        "Cluster embedding shift",
        {"xlabel": "", "ylabel": ""},
    )
    save_panel(
        panel_global_stage_change(cluster_shift_mat=results["cluster_shift_mat"]),
        "line",
        output_dir / "global-stage-change",
        "Aggregate cluster shift",
        {"xlabel": "", "ylabel": "Mean cluster shift", "ymin": 0},
    )
    save_panel(
        panel_e125_e135_module_log2fc(
            e125_e135_cluster_log2fc=results["e125_e135_cluster_log2fc"]
        ),
        "divergent",
        output_dir / "e125-e135-module-log2fc",
        "E13.5 vs E12.5 module score",
        {"ylabel": "log2(E13.5 / E12.5)"},
    )
    save_panel(
        panel_clustermap_gene_sim(gene_sim=results["gene_sim"]),
        "clustermap",
        output_dir / "clustermap-gene-sim",
        "Gene-gene similarity",
        {"xlabel": "", "ylabel": ""},
    )
    save_panel(
        panel_e125_e135_gene_shift(
            e125_e135_gene_shift=results["e125_e135_gene_shift"]
        ),
        "bar",
        output_dir / "e125-e135-gene-shift",
        "E12.5 vs E13.5",
        {"ylabel": "Embedding shift"},
    )
    del results
    for index, stage in enumerate(("E13.5", "E14.5", "E15.5", "E16.5"), start=2):
        results = calculate_maps(
            data_dir,
            view="average",
            resolution=0.5,
            dataset=f"{stage}_E1S1",
            gene="Mt2",
        )
        with (output_dir / f"analysis_{index}.pkl").open("wb") as handle:
            pickle.dump(results, handle, protocol=5)
        suffix = stage.lower().replace(".", "")
        for name, first, second in (("cluster", 0, 1), ("cm", 7, 11)):
            save_panel(
                spatial_panel(
                    results["spatial_xy"],
                    results["cluster_means"][first],
                    results["cluster_means"][second],
                ),
                "spatial",
                output_dir / f"spatial-{name}-{suffix}",
                stage,
                {"channel_labels": [f"Module {first}", f"Module {second}"]},
            )
        if stage == "E13.5":
            save_panel(
                spatial_panel(results["spatial_xy"], results["gene_expr"]),
                "spatial",
                output_dir / "spatial-mt2-e135",
                stage,
                {},
            )
        del results

    gene_maps = [
        ("E12.5", "Tcf7l2", "Tcf7l2 E12.5"),
        ("E13.5", "Tcf7l2", "E13.5"),
        ("E12.5", "Hbb-bt", "Hbb-bt"),
        ("E12.5", "Fabp7", "Fabp7"),
        ("E12.5", "Col2a1", "Col2a1"),
        ("E12.5", "Ttn", "Ttn"),
    ]
    for index, (stage, gene, title) in enumerate(gene_maps, start=6):
        results = calculate_maps(
            data_dir, view="average", resolution=0.5, dataset=f"{stage}_E1S1", gene=gene
        )
        with (output_dir / f"analysis_{index}.pkl").open("wb") as handle:
            pickle.dump(results, handle, protocol=5)
        suffix = stage.lower().replace(".", "")
        save_panel(
            spatial_panel(results["spatial_xy"], results["gene_expr"]),
            "spatial",
            output_dir / f"spatial-{gene.lower()}-{suffix}",
            title,
            {},
        )
        del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "fig2")
