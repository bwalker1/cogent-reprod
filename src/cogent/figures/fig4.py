from pathlib import Path
from cogent import Cogent
from cogent.utils import cluster_means_all
from cogent.analysis import compute_gene_similarity_matrix
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(data_dir, resolution=0.5, visium_sample="6800STDY12499406"):
    data_dir = Path(data_dir)
    int_path = "kidney-carcinoma/models/cogent"
    sc_path = "kidney-carcinoma/models/cogent_sc"
    int_model_file = f"{data_dir}/{int_path}/model.pt"
    int_labels_file = f"{data_dir}/{int_path}/cluster_labels_res0.5.csv"
    sc_model_file = f"{data_dir}/{sc_path}/model.pt"
    visium_file = f"{data_dir}/kidney-carcinoma/raw/visium_interface.h5ad"
    sc_file = f"{data_dir}/kidney-carcinoma/raw/singlecell.h5ad"
    int_model = Cogent.load(Path(int_model_file).parent, device="cpu")
    sc_model = Cogent.load(Path(sc_model_file).parent, device="cpu")
    labels_df = pd.read_csv(int_labels_file)
    adata_v_full = ad.read_h5ad(visium_file)
    adata_v = adata_v_full[
        adata_v_full.obs["sample"].astype(str) == visium_sample
    ].copy()
    ensg = list(int_model._gene_names)
    symbol_map = dict(zip(adata_v.var.index, adata_v.var["feature_name"].astype(str)))
    symbols = [symbol_map.get(e, e) for e in ensg]
    gene_sim = compute_gene_similarity_matrix(int_model, resolution=resolution)
    cluster_means = cluster_means_all(
        int_model, adata_v, resolution=resolution, view="average"
    )
    _xy_raw = (
        adata_v.obsm["X_spatial"]
        if "X_spatial" in adata_v.obsm
        else adata_v.obsm["spatial"]
    )
    spatial_xy = np.asarray(_xy_raw, dtype=np.float32)

    def _shift(m):
        E = m.get_embeddings()
        half = E.shape[-1] // 2
        Emag = E[..., :half]
        norm = Emag / (np.linalg.norm(Emag, axis=-1, keepdims=True) + 1e-09)
        return np.asarray(1.0 - (norm[0] * norm[1]).sum(axis=-1), dtype=float)

    shift_int = _shift(int_model)
    shift_sc = _shift(sc_model)

    def _group_means(sc_path, ensg):
        a = ad.read_h5ad(sc_path)
        disease = a.obs["disease"].astype(str)
        is_c = (disease == "nonpapillary renal cell carcinoma").values
        is_h = (disease == "normal").values
        var_idx = {e: i for (i, e) in enumerate(a.var.index)}
        col = np.array([var_idx.get(e, -1) for e in ensg])
        has = col >= 0
        X = a.X
        cmu = np.asarray(X[is_c].mean(axis=0)).ravel()
        hmu = np.asarray(X[is_h].mean(axis=0)).ravel()
        out_c = np.full(len(ensg), np.nan)
        out_h = np.full(len(ensg), np.nan)
        out_c[has] = cmu[col[has]]
        out_h[has] = hmu[col[has]]
        return {"cancer_mean": out_c, "healthy_mean": out_h}

    sc_means = _group_means(sc_file, ensg)

    def _spatial_gene_expr(adata, gene_symbol):
        mask = adata.var["feature_name"].astype(str) == gene_symbol
        if not mask.any():
            return np.zeros(adata.n_obs)
        ensg_id = adata.var.index[mask][0]
        x = adata[:, ensg_id].X
        return np.asarray(x.toarray() if sp.issparse(x) else x).ravel()

    TOP_SPATIAL_GENES = [
        "CA9",
        "NDUFA4L2",
        "ANGPTL4",
        "EGLN3",
        "NDRG1",
        "BNIP3",
        "GPR183",
        "SLC40A1",
        "KLF2",
    ]
    NEIGHBOR_STORY_GENES = ["GPR183", "SLC40A1", "SEMA4B", "CXCL12", "KLF2"]
    NEIGHBOR_K = 25
    top_spatial_gene_expr = {
        g: np.asarray(_spatial_gene_expr(adata_v, g), dtype=np.float32)
        for g in TOP_SPATIAL_GENES
    }

    def _top_neighbor_indices(normalized_embeddings, query_idx, k):
        similarities = normalized_embeddings @ normalized_embeddings[query_idx]
        similarities[query_idx] = -np.inf
        return np.argsort(-similarities)[:k]

    def _neighbor_replacement(
        symbols, int_model, shift, labels_df, resolution, story_genes, k=25
    ):
        labels = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"]
        ).astype(int)
        label_map = dict(
            zip(labels_df["cluster"].astype(int), labels_df["label"].astype(str))
        )
        gene_symbols = np.asarray(symbols, dtype=str)
        E = int_model.get_embeddings()
        half = E.shape[-1] // 2
        Emag = E[..., :half]
        norm = Emag / (np.linalg.norm(Emag, axis=-1, keepdims=True) + 1e-09)
        cancer_norm = norm[0]
        healthy_norm = norm[1]
        rows = []
        for gi, gene in enumerate(gene_symbols):
            cancer_neighbors = _top_neighbor_indices(cancer_norm, gi, k)
            healthy_neighbors = _top_neighbor_indices(healthy_norm, gi, k)
            shared_n = len(
                set(cancer_neighbors.tolist()) & set(healthy_neighbors.tolist())
            )
            cluster = int(labels[gi])
            rows.append(
                {
                    "gene": str(gene),
                    "shared_n": int(shared_n),
                    "replaced_n": int(k - shared_n),
                    "cluster": cluster,
                    "cluster_label": label_map.get(cluster, "?"),
                    "shift": float(shift[gi]),
                    "cancer_neighbors": ",".join(gene_symbols[cancer_neighbors]),
                    "healthy_neighbors": ",".join(gene_symbols[healthy_neighbors]),
                }
            )
        all_neighbors = pd.DataFrame(rows)
        story_order = {gene: i for (i, gene) in enumerate(story_genes)}
        story = all_neighbors[all_neighbors["gene"].isin(story_genes)].copy()
        story["_story_order"] = story["gene"].map(story_order)
        story = (
            story.sort_values("_story_order")
            .drop(columns=["_story_order"])
            .reset_index(drop=True)
        )
        stable = (
            all_neighbors[~all_neighbors["gene"].isin(story_genes)]
            .sort_values(["replaced_n", "shift", "gene"])
            .head(len(story))
            .reset_index(drop=True)
        )
        return {"neighbor_overlap_df": story, "least_neighbor_overlap_df": stable}

    neighbor_replacement = _neighbor_replacement(
        symbols,
        int_model,
        shift_int,
        labels_df,
        resolution,
        NEIGHBOR_STORY_GENES,
        NEIGHBOR_K,
    )
    neighbor_overlap_df = neighbor_replacement["neighbor_overlap_df"]
    least_neighbor_overlap_df = neighbor_replacement["least_neighbor_overlap_df"]

    def _cluster_jaccard(int_model, sc_model, labels_df, resolution):
        li = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"]
        ).astype(int)
        ls = np.asarray(
            sc_model._leiden_results[resolution]["labels"]["average"]
        ).astype(int)
        ua = sorted(set((int(x) for x in li)))
        ub = sorted(set((int(x) for x in ls)))
        lm = dict(zip(labels_df.cluster.astype(int), labels_df.label))
        rows = []
        for ca in ua:
            ma = li == ca
            (best_j, best_b) = (0.0, ub[0])
            for cb in ub:
                mb = ls == cb
                inter = int((ma & mb).sum())
                uni = int((ma | mb).sum())
                j = inter / uni if uni else 0.0
                if j > best_j:
                    (best_j, best_b) = (j, cb)
            rows.append(
                {
                    "cluster": int(ca),
                    "label": lm.get(int(ca), f"c{int(ca)}"),
                    "n_genes": int(ma.sum()),
                    "best_sconly": int(best_b),
                    "jaccard": float(best_j),
                }
            )
        return pd.DataFrame(rows)

    cluster_jaccard_df = _cluster_jaccard(int_model, sc_model, labels_df, resolution)

    def _cluster_color_index(int_model, labels_df, resolution):
        labs = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"]
        ).astype(int)
        label_map = dict(
            zip(labels_df["cluster"].astype(int), labels_df["label"].astype(str))
        )
        color_index = {}
        for c in labs:
            name = label_map.get(int(c), str(int(c)))
            if name not in color_index:
                color_index[name] = len(color_index)
        return color_index

    cluster_color_index = _cluster_color_index(int_model, labels_df, resolution)
    return {
        "symbols": symbols,
        "gene_sim": gene_sim,
        "cluster_means": cluster_means,
        "spatial_xy": spatial_xy,
        "shift_int": shift_int,
        "shift_sc": shift_sc,
        "sc_means": sc_means,
        "top_spatial_gene_expr": top_spatial_gene_expr,
        "neighbor_overlap_df": neighbor_overlap_df,
        "least_neighbor_overlap_df": least_neighbor_overlap_df,
        "cluster_jaccard_df": cluster_jaccard_df,
        "cluster_color_index": cluster_color_index,
    }


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


def panel_shared_umap(data_dir):
    from pathlib import Path
    import pandas as pd
    from cogent import Cogent

    model_dir = Path(data_dir / "kidney-carcinoma/models/cogent")
    m = Cogent.load(model_dir, device="cpu")
    coords = m._umap_coords["average"]
    labs = [int(label) for label in m._leiden_results[0.5]["labels"]["average"]]
    labels_df = pd.read_csv(model_dir / "cluster_labels_res0.5.csv")
    label_map = dict(zip(labels_df["cluster"], labels_df["label"]))

    def fmt(c):
        return label_map[c] if c in label_map else str(c)

    x = coords[:, 0]
    y = coords[:, 1]
    values = [fmt(label) for label in labs]
    type = "categorical"
    _meta = {
        "cluster_sizes": {
            fmt(int(r["cluster"])): int(r["n_genes"]) for (_, r) in labels_df.iterrows()
        }
    }
    return {"x": x, "y": y, "values": values, "type": type, "_meta": _meta}


def panel_scatter_shift_dmean(symbols, shift_int, sc_means):
    import numpy as np

    abs_dm = np.abs(sc_means["cancer_mean"] - sc_means["healthy_mean"])
    shift_q75 = float(np.nanquantile(shift_int, 0.75))
    dm_q75 = float(np.nanquantile(abs_dm, 0.75))

    def _cat(s, d):
        if np.isnan(d):
            return "stable"
        if s >= shift_q75 and d >= dm_q75:
            return "DE + shift"
        if s >= shift_q75:
            return "shift only"
        if d >= dm_q75:
            return "DE only"
        return "stable"

    highlight = {
        "CA9",
        "NDUFA4L2",
        "EGLN3",
        "HILPDA",
        "ANGPTL4",
        "GPR183",
        "SLC40A1",
        "SEMA4B",
        "CXCL12",
        "KLF2",
        "GPNMB",
    }
    points = []
    for i, g in enumerate(symbols):
        dm = abs_dm[i]
        if np.isnan(dm):
            continue
        points.append(
            {
                "x": float(dm),
                "y": float(shift_int[i]),
                "category": _cat(float(shift_int[i]), float(dm)),
                "label": g if g in highlight else "",
            }
        )
    categories = [
        {"name": "shift only", "color": "#e07b54"},
        {"name": "DE + shift", "color": "#7b54e0"},
        {"name": "DE only", "color": "#4c8fbf"},
        {"name": "stable", "color": "#cccccc"},
    ]
    log_scale = {"x": False, "y": False}
    return {"points": points, "categories": categories, "log_scale": log_scale}


def panel_bar_tumor_fc(symbols, sc_means):
    import numpy as np

    GENES = ["CA9", "NDUFA4L2", "ANGPTL4", "EGLN3", "HILPDA", "BNIP3", "NDRG1"]
    bars = []
    for g in GENES:
        if g not in symbols:
            continue
        i = symbols.index(g)
        c = sc_means["cancer_mean"][i] + 1e-06
        h = sc_means["healthy_mean"][i] + 1e-06
        bars.append({"label": g, "value": float(np.log2(c / h))})
    return {"bars": bars}


def panel_bar_tumor_shift(symbols, shift_int):
    GENES = ["CA9", "NDUFA4L2", "ANGPTL4", "EGLN3", "HILPDA", "BNIP3", "NDRG1"]
    bars = []
    for g in GENES:
        if g not in symbols:
            continue
        i = symbols.index(g)
        bars.append({"label": g, "value": float(shift_int[i])})
    bars = sorted(bars, key=lambda b: b["value"], reverse=True)
    return {"bars": bars}


def panel_bar_cogent_unique(symbols, shift_int):
    GENES = [
        "GPR183",
        "SLC40A1",
        "SEMA4B",
        "CXCL12",
        "KLF2",
        "GPNMB",
        "SCAMP5",
        "TXNDC5",
    ]
    bars = []
    for g in GENES:
        if g not in symbols:
            continue
        i = symbols.index(g)
        bars.append({"label": g, "value": float(shift_int[i])})
    bars = sorted(bars, key=lambda b: b["value"], reverse=True)
    return {"bars": bars}


def panel_bar_neighbor_replacement(neighbor_overlap_df, least_neighbor_overlap_df):
    high_bars = [
        {"label": str(r["gene"]), "value": float(r["replaced_n"])}
        for (_, r) in neighbor_overlap_df.sort_values("replaced_n", ascending=False)
        .head(5)
        .iterrows()
    ]
    low_bars = [
        {"label": str(r["gene"]), "value": float(r["replaced_n"])}
        for (_, r) in least_neighbor_overlap_df.sort_values(
            "replaced_n", ascending=False
        )
        .head(5)
        .iterrows()
    ]
    bars = high_bars + [{"label": " ", "value": 0.0, "color": "transparent"}] + low_bars
    return {"bars": bars}


def panel_bar_jaccard_recovery(cluster_jaccard_df, cluster_color_index):
    df = cluster_jaccard_df.sort_values("jaccard", ascending=False)
    bars = []
    for _, r in df.iterrows():
        if int(r["n_genes"]) < 10:
            continue
        name = str(r["label"])
        label = f"c{int(r['cluster'])}: {name[:22]}"
        bars.append(
            {
                "label": label,
                "value": float(r["jaccard"]),
                "colorIndex": cluster_color_index.get(name, 0),
            }
        )
    return {"bars": bars}


def panel_scatter_int_vs_sc(symbols, shift_int, shift_sc):
    highlight = {
        "GPR183",
        "SLC40A1",
        "GPNMB",
        "SCAMP5",
        "TXNDC5",
        "CXCL12",
        "CD9",
        "SCD",
        "SEMA4B",
        "IDH1",
        "KLF2",
        "S100A6",
        "MT-ND1",
    }
    points = []
    for i, g in enumerate(symbols):
        points.append(
            {
                "x": float(shift_sc[i]),
                "y": float(shift_int[i]),
                "category": "highlight" if g in highlight else "other",
                "label": g if g in highlight else "",
            }
        )
    categories = [
        {"name": "highlight", "color": "#e07b54"},
        {"name": "other", "color": "#cccccc"},
    ]
    log_scale = {"x": False, "y": False}
    return {"points": points, "categories": categories, "log_scale": log_scale}


def panel_kde_shift(shift_int, shift_sc):
    groups = [
        {"label": "integrated", "values": [float(s) for s in shift_int]},
        {"label": "sconly", "values": [float(s) for s in shift_sc]},
    ]
    return {"groups": groups}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5, visium_sample="6800STDY12499406")
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_clustermap_gene_sim(gene_sim=results["gene_sim"]),
        "clustermap",
        output_dir / "clustermap-gene-sim",
        "Gene-gene cosine similarity",
        {"xlabel": "", "ylabel": ""},
    )
    save_panel(
        panel_shared_umap(data_dir),
        "umap",
        output_dir / "shared-umap",
        "Shared UMAP",
        {},
    )
    save_panel(
        panel_scatter_shift_dmean(
            symbols=results["symbols"],
            shift_int=results["shift_int"],
            sc_means=results["sc_means"],
        ),
        "scatter",
        output_dir / "scatter-shift-dmean",
        "Shift vs |Δmean| (singlecell)",
        {"xlabel": "|Δmean|", "ylabel": "Cosine shift"},
    )
    save_panel(
        panel_bar_tumor_fc(symbols=results["symbols"], sc_means=results["sc_means"]),
        "bar",
        output_dir / "bar-tumor-fc",
        "Tumor genes",
        {"ylabel": "log₂ FC"},
    )
    save_panel(
        panel_bar_tumor_shift(
            symbols=results["symbols"], shift_int=results["shift_int"]
        ),
        "bar",
        output_dir / "bar-tumor-shift",
        "Tumor Genes",
        {"ylabel": "Shift"},
    )
    save_panel(
        panel_bar_cogent_unique(
            symbols=results["symbols"], shift_int=results["shift_int"]
        ),
        "bar",
        output_dir / "bar-cogent-unique",
        "Embedding shift genes",
        {"ylabel": "Shift"},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][10],
            results["cluster_means"][14],
        ),
        "spatial",
        output_dir / "spatial-c10-c14",
        "Tumor + MHC-II (c10, c14)",
        {"channel_labels": ["Module 10", "Module 14"]},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][4],
            results["cluster_means"][8],
        ),
        "spatial",
        output_dir / "spatial-c4-c8",
        "T cells + TAMs (c4, c8)",
        {"channel_labels": ["Module 4", "Module 8"]},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][2],
            results["cluster_means"][17],
        ),
        "spatial",
        output_dir / "spatial-c2-c17",
        "Vascular + DCs (c2, c17)",
        {"channel_labels": ["Module 2", "Module 17"]},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"],
            results["cluster_means"][7],
            results["cluster_means"][9],
        ),
        "spatial",
        output_dir / "spatial-c7-c9",
        "CAFs + stroma (c7, c9)",
        {"channel_labels": ["Module 7", "Module 9"]},
    )
    save_panel(
        panel_bar_neighbor_replacement(
            neighbor_overlap_df=results["neighbor_overlap_df"],
            least_neighbor_overlap_df=results["least_neighbor_overlap_df"],
        ),
        "bar",
        output_dir / "bar-neighbor-replacement",
        "Neighborhood Replacement",
        {"ylabel": "neighbors changed", "ymax": 25},
    )
    save_panel(
        panel_bar_jaccard_recovery(
            cluster_jaccard_df=results["cluster_jaccard_df"],
            cluster_color_index=results["cluster_color_index"],
        ),
        "bar",
        output_dir / "bar-jaccard-recovery",
        "Module recovery vs sconly (Jaccard)",
        {"ylabel": "Jaccard score"},
    )
    save_panel(
        panel_scatter_int_vs_sc(
            symbols=results["symbols"],
            shift_int=results["shift_int"],
            shift_sc=results["shift_sc"],
        ),
        "scatter",
        output_dir / "scatter-int-vs-sc",
        "Shift: integrated vs sc-only",
        {"xlabel": "Shift (sconly)", "ylabel": "Shift (integrated)"},
    )
    save_panel(
        panel_kde_shift(shift_int=results["shift_int"], shift_sc=results["shift_sc"]),
        "kde",
        output_dir / "kde-shift",
        "Shift distributions: integrated vs sconly",
        {"xlabel": "Shift", "xmax": 0.75, "xmin": 0},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["top_spatial_gene_expr"]["CA9"]),
        "spatial",
        output_dir / "spatial-d-top1-ca9",
        "CA9",
        {},
    )
    save_panel(
        spatial_panel(
            results["spatial_xy"], results["top_spatial_gene_expr"]["NDUFA4L2"]
        ),
        "spatial",
        output_dir / "spatial-d-top2-ndufa4l2",
        "NDUFA4L2",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["top_spatial_gene_expr"]["EGLN3"]),
        "spatial",
        output_dir / "spatial-e-top1-egln3",
        "EGLN3",
        {},
    )
    save_panel(
        spatial_panel(results["spatial_xy"], results["top_spatial_gene_expr"]["NDRG1"]),
        "spatial",
        output_dir / "spatial-e-top2-ndrg1",
        "NDRG1",
        {},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "fig4")
