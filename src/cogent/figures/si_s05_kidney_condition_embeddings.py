from pathlib import Path
from cogent import Cogent
from cogent.utils import cluster_means_all
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from umap import UMAP
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(data_dir, resolution=0.5, visium_sample="6800STDY12499406"):
    data_dir = Path(data_dir)
    int_model_file = data_dir / "kidney-carcinoma/models/cogent/model.pt"
    int_labels_file = (
        data_dir / "kidney-carcinoma/models/cogent/cluster_labels_res0.5.csv"
    )
    visium_file = data_dir / "kidney-carcinoma/raw/visium_interface.h5ad"

    def load_int_model():
        return Cogent.load(Path(int_model_file).parent, device="cpu")

    def embedding_component(model):
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        return emb[..., :half]

    def model_shift(model):
        emb = embedding_component(model)
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        return np.asarray(1.0 - np.sum(norm[0] * norm[1], axis=1), dtype=np.float32)

    def symbols_for(model):
        a = ad.read_h5ad(visium_file, backed="r")
        if "feature_name" in a.var:
            symbol_map = dict(
                zip(a.var.index.astype(str), a.var["feature_name"].astype(str))
            )
        else:
            symbol_map = {}
        out = [symbol_map.get(str(g), str(g)) for g in model._gene_names]
        a.file.close()
        return out

    def build_condition_umap_payloads(model, labels_path, left_label, right_label):
        emb = embedding_component(model)
        (n_groups, n_genes, dim) = emb.shape
        coords = (
            UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            .fit_transform(emb.reshape(n_groups * n_genes, dim))
            .astype(np.float32)
        )
        coords = coords.reshape(n_groups, n_genes, 2)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = pd.read_csv(labels_path)
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        values = [f"c{int(c)}: {label_map.get(int(c), 'unlabeled')}" for c in labs]
        return {
            left_label: {
                "x": coords[0, :, 0],
                "y": coords[0, :, 1],
                "values": values,
                "type": "categorical",
            },
            right_label: {
                "x": coords[1, :, 0],
                "y": coords[1, :, 1],
                "values": values,
                "type": "categorical",
            },
        }

    def module_centroid_shift(model, labels_path):
        emb = embedding_component(model)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = pd.read_csv(labels_path)
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
            if len(idx) == 0:
                continue
            c0 = emb[0, idx, :].mean(axis=0)
            c1 = emb[1, idx, :].mean(axis=0)
            c0 = c0 / (np.linalg.norm(c0) + 1e-09)
            c1 = c1 / (np.linalg.norm(c1) + 1e-09)
            rows.append(
                {
                    "cluster": int(cluster),
                    "label": label_map.get(int(cluster), "unlabeled"),
                    "n_genes": int(len(idx)),
                    "shift": float(1.0 - np.dot(c0, c1)),
                }
            )
        return pd.DataFrame(rows)

    def displacement_df(model, labels_path, symbols, n_arrows=180):
        payload = build_condition_umap_payloads(
            model, labels_path, "tumor", "reference"
        )
        shift = model_shift(model)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        order = np.argsort(-shift)[:n_arrows]
        return pd.DataFrame(
            {
                "gene": [symbols[i] for i in order],
                "cluster": labs[order].astype(int),
                "x": payload["reference"]["x"][order],
                "y": payload["reference"]["y"][order],
                "u": payload["tumor"]["x"][order] - payload["reference"]["x"][order],
                "v": payload["tumor"]["y"][order] - payload["reference"]["y"][order],
                "shift": shift[order],
            }
        )

    int_model = load_int_model()
    kidney_symbols = symbols_for(int_model)
    HIGHLIGHT_GENES = [
        "CA9",
        "NDUFA4L2",
        "ANGPTL4",
        "EGLN3",
        "GPR183",
        "SLC40A1",
        "GPNMB",
        "KLF2",
        "CXCL12",
        "SEMA4B",
    ]

    def build_highlight_gene_df():
        shift = model_shift(int_model)
        labs = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        coords = np.asarray(int_model._umap_coords["average"], dtype=np.float32)
        rows = []
        for gene in HIGHLIGHT_GENES:
            if gene not in kidney_symbols:
                continue
            idx = kidney_symbols.index(gene)
            rows.append(
                {
                    "gene": gene,
                    "cluster": int(labs[idx]),
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "shift": float(shift[idx]),
                }
            )
        return pd.DataFrame(rows)

    condition_displacement_df = displacement_df(
        int_model, int_labels_file, kidney_symbols
    )
    module_centroid_shift_df = module_centroid_shift(int_model, int_labels_file)
    highlight_gene_df = build_highlight_gene_df()

    def load_visium_sample():
        a = ad.read_h5ad(visium_file)
        return a[a.obs["sample"].astype(str) == visium_sample].copy()

    def build_spatial_xy(a):
        key = "X_spatial" if "X_spatial" in a.obsm else "spatial"
        return np.asarray(a.obsm[key], dtype=np.float32)

    def build_spatial_cluster_means(a):
        return {
            int(k): np.asarray(v, dtype=np.float32)
            for (k, v) in cluster_means_all(
                int_model, a, view="average", resolution=resolution
            ).items()
        }

    def build_mask_payload(xy, cm):
        tumor = np.asarray(cm[10], dtype=np.float32)
        threshold = float(np.quantile(tumor, 0.75))
        mask = tumor >= threshold
        return {
            "values": mask.astype(np.float32),
            "threshold": threshold,
            "cluster": 10,
        }

    def build_inside_outside(cm, mask_payload):
        mask = np.asarray(mask_payload["values"]) > 0
        labels = {
            2: "vascular/stroma",
            17: "dendritic",
            7: "CAF",
            9: "contractile",
            10: "tumor",
        }
        rows = []
        for cluster, label in labels.items():
            vals = np.asarray(cm[int(cluster)], dtype=np.float32)
            for region, keep in [("inside", mask), ("outside", ~mask)]:
                sample = vals[keep]
                if len(sample) > 800:
                    rng = np.random.default_rng(
                        cluster + (0 if region == "inside" else 100)
                    )
                    sample = rng.choice(sample, size=800, replace=False)
                for value in sample:
                    rows.append(
                        {
                            "cluster": int(cluster),
                            "label": label,
                            "region": region,
                            "score": float(value),
                        }
                    )
        return pd.DataFrame(rows)

    def build_distance_profile(xy, cm, mask_payload):
        mask = np.asarray(mask_payload["values"]) > 0
        if mask.sum() == 0 or (~mask).sum() == 0:
            dist = np.zeros(len(mask), dtype=np.float32)
        else:
            inside_tree = cKDTree(xy[mask])
            outside_tree = cKDTree(xy[~mask])
            d_inside = inside_tree.query(xy, k=1)[0]
            d_outside = outside_tree.query(xy, k=1)[0]
            dist = d_inside.astype(np.float32)
            dist[mask] = -d_outside[mask]
        labels = {
            2: "vascular/stroma",
            17: "dendritic",
            7: "CAF",
            9: "contractile",
            10: "tumor",
        }
        finite = np.isfinite(dist)
        edges = np.quantile(dist[finite], np.linspace(0, 1, 9))
        edges = np.unique(edges)
        rows = []
        for cluster, label in labels.items():
            vals = np.asarray(cm[int(cluster)], dtype=np.float32)
            for i in range(len(edges) - 1):
                keep = (dist >= edges[i]) & (dist <= edges[i + 1])
                if keep.sum() == 0:
                    continue
                rows.append(
                    {
                        "cluster": int(cluster),
                        "label": label,
                        "bin": int(i),
                        "distance": float((edges[i] + edges[i + 1]) / 2),
                        "mean_score": float(np.mean(vals[keep])),
                    }
                )
        return pd.DataFrame(rows)

    adata_v = load_visium_sample()
    spatial_xy = build_spatial_xy(adata_v)
    cluster_means = build_spatial_cluster_means(adata_v)
    tumor_mask_payload = build_mask_payload(spatial_xy, cluster_means)
    outside_inside_df = build_inside_outside(cluster_means, tumor_mask_payload)
    distance_profile_df = build_distance_profile(
        spatial_xy, cluster_means, tumor_mask_payload
    )
    return {
        "condition_displacement_df": condition_displacement_df,
        "module_centroid_shift_df": module_centroid_shift_df,
        "highlight_gene_df": highlight_gene_df,
        "spatial_xy": spatial_xy,
        "cluster_means": cluster_means,
        "tumor_mask_payload": tumor_mask_payload,
        "outside_inside_df": outside_inside_df,
        "distance_profile_df": distance_profile_df,
    }


def panel_s4_displacement(condition_displacement_df):
    df = condition_displacement_df
    x = df["x"].astype(float).tolist()
    y = df["y"].astype(float).tolist()
    u = df["u"].astype(float).tolist()
    v = df["v"].astype(float).tolist()
    magnitude = df["shift"].astype(float).tolist()
    return {"x": x, "y": y, "u": u, "v": v, "magnitude": magnitude}


def panel_s4_centroid_shift(module_centroid_shift_df):
    df = module_centroid_shift_df[
        module_centroid_shift_df["n_genes"] >= 10
    ].sort_values("shift", ascending=False)
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["shift"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s4_highlight(highlight_gene_df):
    points = [
        {
            "x": float(r["shift"]),
            "y": float(r["cluster"]),
            "label": str(r["gene"]),
            "category": "highlight",
        }
        for (_, r) in highlight_gene_df.iterrows()
    ]
    categories = [{"name": "highlight", "color": "#c45a40"}]
    return {"points": points, "categories": categories}


def panel_s5_mask(spatial_xy, tumor_mask_payload):
    return {
        "x": spatial_xy[:, 0],
        "y": -spatial_xy[:, 1],
        "values": tumor_mask_payload["values"],
        "type": "continuous",
        "_meta": {
            "threshold": tumor_mask_payload["threshold"],
            "cluster": tumor_mask_payload["cluster"],
        },
    }


def panel_s5_inside_outside(outside_inside_df):
    df = outside_inside_df[outside_inside_df["cluster"].isin([2, 17, 7, 9])]
    groups = []
    for (label, region), sub in df.groupby(["label", "region"], sort=False):
        groups.append(
            {
                "label": f"{label}\n{region}",
                "values": sub["score"].astype(float).tolist(),
            }
        )
    return {"groups": groups}


def panel_s5_distance(distance_profile_df):
    series = []
    for label, sub in distance_profile_df[
        distance_profile_df["cluster"].isin([2, 17, 7, 9])
    ].groupby("label", sort=False):
        points = [
            {"x": float(r["distance"]), "y": float(r["mean_score"])}
            for (_, r) in sub.sort_values("distance").iterrows()
        ]
        series.append({"name": str(label), "points": points})
    return {"series": series}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5, visium_sample="6800STDY12499406")
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s4_displacement(
            condition_displacement_df=results["condition_displacement_df"]
        ),
        "quiver",
        output_dir / "s4-displacement",
        "Tumor-reference displacement",
        {},
    )
    save_panel(
        panel_s4_centroid_shift(
            module_centroid_shift_df=results["module_centroid_shift_df"]
        ),
        "bar",
        output_dir / "s4-centroid-shift",
        "Module centroid shift",
        {"ylabel": "cosine distance"},
    )
    save_panel(
        panel_s4_highlight(highlight_gene_df=results["highlight_gene_df"]),
        "scatter",
        output_dir / "s4-highlight",
        "Highlighted gene shifts",
        {"xlabel": "tumor-reference shift", "ylabel": "module"},
    )
    save_panel(
        panel_s5_mask(
            spatial_xy=results["spatial_xy"],
            tumor_mask_payload=results["tumor_mask_payload"],
        ),
        "spatial",
        output_dir / "s5-mask",
        "Tumor-core proxy mask",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means"][int(2)]),
        "spatial",
        output_dir / "s5-vascular",
        "Endothelium/vascular stroma",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means"][int(17)]),
        "spatial",
        output_dir / "s5-dendritic",
        "Dendritic-cell module",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means"][int(7)]),
        "spatial",
        output_dir / "s5-caf",
        "CAF/myofibroblast module",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means"][int(9)]),
        "spatial",
        output_dir / "s5-contractile",
        "Contractile stromal module",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["cluster_means"][int(8)]),
        "spatial",
        output_dir / "s5-macrophage",
        "Tumor-associated macrophage module",
        {},
    )
    save_panel(
        panel_s5_inside_outside(outside_inside_df=results["outside_inside_df"]),
        "boxplot",
        output_dir / "s5-inside-outside",
        "Inside/outside tumor mask",
        {"ylabel": "module score"},
    )
    save_panel(
        panel_s5_distance(distance_profile_df=results["distance_profile_df"]),
        "line",
        output_dir / "s5-distance",
        "Distance from tumor-core proxy",
        {"xlabel": "signed distance", "ylabel": "mean module score"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s05_kidney_condition_embeddings")
