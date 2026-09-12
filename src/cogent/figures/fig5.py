from pathlib import Path
from cogent import Cogent
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from umap import UMAP
from cogent.figures.plotting import save_panel


def calculate(data_dir):
    data_dir = Path(data_dir)
    model_file = data_dir / "visium_crc/models/cogent/model.pt"
    labels_file = data_dir / "visium_crc/models/cogent/cluster_labels_res0.5.csv"
    p1_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P1_square_008um.h5ad"
    )
    crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P2_square_008um.h5ad"
    )
    p5_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P5_square_008um.h5ad"
    )
    p3_nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P3_square_008um.h5ad"
    )
    nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P5_square_008um.h5ad"
    )
    all_sample_files = [p1_crc_file, crc_file, p5_crc_file, p3_nat_file, nat_file]
    SPATIAL_CLUSTERS = [7, 15, 4]
    SPATIAL_GENES = ["EMP1", "UGCG", "ADM"]
    STORY_GENES = [
        "EMP1",
        "DHCR24",
        "DAB2",
        "LEFTY1",
        "CPNE1",
        "CEACAM6",
        "CEACAM5",
        "TFF3",
        "AQP1",
        "SLC7A5",
    ]
    NEIGHBOR_K = 25
    EMP1_NEIGHBOR_K = 10
    _model_cache = None

    def load_model(model_path):
        nonlocal _model_cache
        if _model_cache is None:
            _model_cache = Cogent.load(Path(model_path).parent, device="cpu")
        return _model_cache

    def cluster_labels_df(path):
        return pd.read_csv(path)

    def label_lookup(path):
        df = cluster_labels_df(path)
        return dict(zip(df["cluster"].astype(int), df["label"].astype(str)))

    def format_cluster(cluster, labels):
        return f"{cluster}: {labels[cluster]}" if cluster in labels else str(cluster)

    def build_umap_avg(model_path):
        model = load_model(model_path)
        coords = np.asarray(model._umap_coords["average"], dtype=np.float32)
        return np.asarray(coords[:, :2], dtype=np.float32)

    def build_umap_avg_values(model_path, labels_path):
        model = load_model(model_path)
        labels = np.asarray(model._leiden_results[0.5]["labels"]["average"], dtype=int)
        lookup = label_lookup(labels_path)
        return [format_cluster(int(c), lookup) for c in labels]

    def build_umap_avg_meta(labels_path):
        df = cluster_labels_df(labels_path)
        lookup = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        return {
            "cluster_sizes": {
                format_cluster(int(r["cluster"]), lookup): int(r["n_genes"])
                for (_, r) in df.iterrows()
            }
        }

    def build_umap_shared(model_path):
        model = load_model(model_path)
        embeddings = np.asarray(model.get_embeddings(), dtype=np.float32)
        half_dim = embeddings.shape[-1] // 2
        first_half = embeddings[:, :, :half_dim].reshape(-1, half_dim)
        coords = (
            UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="cosine",
                init="spectral",
                random_state=42,
            )
            .fit_transform(first_half)
            .astype(np.float32)
        )
        return np.asarray(coords[:, :2], dtype=np.float32)

    def build_umap_shared_values(model_path):
        model = load_model(model_path)
        group_ids = model._group_ids or [
            str(i) for i in range(model._embeddings.shape[0])
        ]
        n_genes = int(model._embeddings.shape[1])
        values = []
        for gid in group_ids:
            values.extend([str(gid)] * n_genes)
        return values

    def load_normalized_sample(sample_path):
        a = ad.read_h5ad(sample_path)
        a = a[a.obs["in_tissue"] == 1].copy()
        sc_pp = __import__("scanpy").pp
        sc_pp.normalize_total(a, target_sum=10000.0)
        sc_pp.log1p(a)
        return a

    def cluster_genes(model, cluster):
        labels = np.asarray(model._leiden_results[0.5]["labels"]["average"], dtype=int)
        return [model._gene_names[i] for i in np.where(labels == cluster)[0]]

    def spatial_raw_payload(a, model):
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        var_idx = {g: i for (i, g) in enumerate(a.var_names)}
        out = {
            "x": np.asarray(coords[:, 0], dtype=np.float32),
            "y": np.asarray(-coords[:, 1], dtype=np.float32),
        }
        for cluster in SPATIAL_CLUSTERS:
            genes = cluster_genes(model, cluster)
            idxs = [var_idx[g] for g in genes if g in var_idx]
            sub = a.X[:, idxs]
            score = (
                np.asarray(sub.mean(axis=1)).ravel()
                if sp.issparse(sub)
                else sub.mean(axis=1)
            )
            out[f"cluster_{cluster}"] = np.asarray(score, dtype=np.float32)
        for gene in SPATIAL_GENES:
            if gene not in var_idx:
                raise ValueError(f"{gene} not in var_names")
            col = a.X[:, var_idx[gene]]
            expr = (
                np.asarray(col.todense()).ravel()
                if sp.issparse(col)
                else np.asarray(col).ravel()
            )
            out[gene] = np.asarray(expr, dtype=np.float32)
        return out

    def build_spatial_payload(sample_path, model_path):
        model = load_model(model_path)
        return spatial_raw_payload(load_normalized_sample(sample_path), model)

    def build_spatial_meta(sample_path, model_path):
        model = load_model(model_path)
        a = ad.read_h5ad(sample_path, backed="r")
        var_names = set(a.var_names.astype(str))
        out = {}
        for cluster in SPATIAL_CLUSTERS:
            genes = cluster_genes(model, cluster)
            out[f"cluster_{cluster}"] = {
                "cluster": int(cluster),
                "n_genes": int(sum((1 for g in genes if g in var_names))),
            }
        for gene in SPATIAL_GENES:
            if gene not in var_names:
                raise ValueError(f"{gene} not in var_names")
            out[gene] = {"gene": gene}
        a.file.close()
        return out

    def parse_visium_sample(sample_path):
        parts = Path(sample_path).stem.split("_")
        group = "CRC" if parts[4] == "Cancer" else "NAT"
        return (f"{parts[5]}_{group}", parts[5], group)

    def top_neighbor_indices(normalized_embeddings, query_idx, k):
        similarities = normalized_embeddings @ normalized_embeddings[query_idx]
        similarities[query_idx] = -np.inf
        return np.argsort(-similarities)[:k]

    def build_fig5_analysis(model_path, labels_path, sample_paths):
        model = load_model(model_path)
        labels_df = pd.read_csv(labels_path)
        label_map = dict(
            zip(labels_df["cluster"].astype(int), labels_df["label"].astype(str))
        )
        labels = np.asarray(model._leiden_results[0.5]["labels"]["average"], dtype=int)
        gene_names = np.asarray(model._gene_names, dtype=str)
        clusters = labels_df["cluster"].astype(int).tolist()
        sample_rows = []
        mean_expr = np.zeros(len(gene_names), dtype=np.float64)
        crc_expr = np.zeros(len(gene_names), dtype=np.float64)
        nat_expr = np.zeros(len(gene_names), dtype=np.float64)
        crc_n = 0
        nat_n = 0
        for sample_path in sample_paths:
            (sample, patient, group) = parse_visium_sample(sample_path)
            a = load_normalized_sample(sample_path)
            var_idx = {g: i for (i, g) in enumerate(a.var_names)}
            row = {
                "sample": sample,
                "patient": patient,
                "group": group,
                "n_spots": int(a.n_obs),
            }
            for cluster in clusters:
                members = gene_names[labels == cluster]
                idxs = [var_idx[g] for g in members if g in var_idx]
                sub = a.X[:, idxs]
                score = (
                    np.asarray(sub.mean(axis=1)).ravel()
                    if sp.issparse(sub)
                    else sub.mean(axis=1)
                )
                row[f"c{cluster}"] = float(score.mean())
            ordered = np.array([var_idx.get(g, -1) for g in gene_names])
            mask = ordered >= 0
            sub = a.X[:, ordered[mask]]
            per_gene = (
                np.asarray(sub.mean(axis=0)).ravel()
                if sp.issparse(sub)
                else sub.mean(axis=0)
            )
            full = np.zeros(len(gene_names), dtype=np.float64)
            full[mask] = per_gene
            mean_expr += full
            if group == "CRC":
                crc_expr += full
                crc_n += 1
            else:
                nat_expr += full
                nat_n += 1
            sample_rows.append(row)
        scores_df = pd.DataFrame(sample_rows)
        rows = []
        for cluster in clusters:
            col = f"c{cluster}"
            crc = scores_df.loc[scores_df["group"] == "CRC", col].mean()
            nat = scores_df.loc[scores_df["group"] == "NAT", col].mean()
            rows.append(
                {
                    "cluster": int(cluster),
                    "label": label_map.get(int(cluster), "?"),
                    "n_genes": int((labels == cluster).sum()),
                    "crc_mean": float(crc),
                    "nat_mean": float(nat),
                    "log2_crc_over_nat": float(
                        np.log2((crc + 0.0001) / (nat + 0.0001))
                    ),
                }
            )
        log2fc_df = (
            pd.DataFrame(rows)
            .sort_values("log2_crc_over_nat", ascending=False)
            .reset_index(drop=True)
        )
        mean_expr /= len(sample_paths)
        crc_expr /= max(crc_n, 1)
        nat_expr /= max(nat_n, 1)
        delta_mean = crc_expr - nat_expr
        embeddings = np.asarray(model.get_embeddings(), dtype=np.float32)
        half_dim = embeddings.shape[-1] // 2
        emb = embeddings[:, :, :half_dim]
        group_ids = [str(g) for g in model._group_ids or []]
        if "CRC" in group_ids and "NAT" in group_ids:
            crc_idx = group_ids.index("CRC")
            nat_idx = group_ids.index("NAT")
        else:
            crc_idx = 0
            nat_idx = 1
        crc_emb = emb[crc_idx]
        nat_emb = emb[nat_idx]
        shift_emb = emb[[crc_idx, nat_idx]]
        shift = np.empty(len(gene_names), dtype=np.float32)
        for gi in range(len(gene_names)):
            shift[gi] = pdist(shift_emb[:, gi, :], metric="cosine").mean()
        shift_hi = shift >= np.percentile(shift, 90)
        dm_hi = np.abs(delta_mean) >= np.percentile(np.abs(delta_mean), 75)
        category = np.full(len(gene_names), "low", dtype=object)
        category[shift_hi & dm_hi] = "1_DE_and_shift"
        category[shift_hi & ~dm_hi] = "2_context_only"
        category[~shift_hi & dm_hi] = "3_expr_only"
        shift_df = pd.DataFrame(
            {
                "gene": gene_names,
                "shift": shift,
                "cluster": labels,
                "cluster_label": [label_map.get(int(c), "?") for c in labels],
                "mean_expr": mean_expr,
                "delta_mean_crc_nat": delta_mean,
                "abs_delta": np.abs(delta_mean),
                "category": category,
            }
        ).sort_values("shift", ascending=False)
        crc_norm = crc_emb / np.clip(
            np.linalg.norm(crc_emb, axis=1, keepdims=True), 1e-12, None
        )
        nat_norm = nat_emb / np.clip(
            np.linalg.norm(nat_emb, axis=1, keepdims=True), 1e-12, None
        )
        neighbor_rows = []
        for gi, gene in enumerate(gene_names):
            crc_neighbors = top_neighbor_indices(crc_norm, gi, NEIGHBOR_K)
            nat_neighbors = top_neighbor_indices(nat_norm, gi, NEIGHBOR_K)
            shared_n = len(set(crc_neighbors) & set(nat_neighbors))
            cluster = int(labels[gi])
            neighbor_rows.append(
                {
                    "gene": gene,
                    "shared_n": int(shared_n),
                    "replaced_n": int(NEIGHBOR_K - shared_n),
                    "cluster": cluster,
                    "cluster_label": label_map.get(cluster, "?"),
                    "shift": float(shift[gi]),
                    "delta_mean_crc_nat": float(delta_mean[gi]),
                    "crc_neighbors": ",".join(gene_names[crc_neighbors]),
                    "nat_neighbors": ",".join(gene_names[nat_neighbors]),
                }
            )
        all_neighbor_overlap_df = pd.DataFrame(neighbor_rows)
        story_order = {gene: i for (i, gene) in enumerate(STORY_GENES)}
        neighbor_overlap_df = all_neighbor_overlap_df[
            all_neighbor_overlap_df["gene"].isin(STORY_GENES)
        ].copy()
        neighbor_overlap_df["_story_order"] = neighbor_overlap_df["gene"].map(
            story_order
        )
        neighbor_overlap_df = (
            neighbor_overlap_df.sort_values("_story_order")
            .drop(columns=["_story_order"])
            .reset_index(drop=True)
        )
        least_neighbor_overlap_df = (
            all_neighbor_overlap_df[~all_neighbor_overlap_df["gene"].isin(STORY_GENES)]
            .sort_values(["replaced_n", "shift", "gene"])
            .head(len(neighbor_overlap_df))
            .reset_index(drop=True)
        )
        emp1_idx = {g: i for (i, g) in enumerate(gene_names)}["EMP1"]
        emp1_neighbor_rows = []
        for condition, normalized in [("CRC", crc_norm), ("NAT", nat_norm)]:
            similarities = normalized @ normalized[emp1_idx]
            for rank, gi in enumerate(
                top_neighbor_indices(normalized, emp1_idx, EMP1_NEIGHBOR_K), start=1
            ):
                emp1_neighbor_rows.append(
                    {
                        "condition": condition,
                        "rank": rank,
                        "gene": str(gene_names[gi]),
                        "similarity": float(similarities[gi]),
                        "cluster": int(labels[gi]),
                        "cluster_label": label_map.get(int(labels[gi]), "?"),
                    }
                )
        emp1_neighbors_df = pd.DataFrame(emp1_neighbor_rows)
        return {
            "log2fc_df": log2fc_df,
            "shift_df": shift_df,
            "neighbor_overlap_df": neighbor_overlap_df,
            "least_neighbor_overlap_df": least_neighbor_overlap_df,
            "emp1_neighbors_df": emp1_neighbors_df,
        }

    def add_shift_labels(df):
        df = df.copy()
        finite = np.isfinite(df["mean_expr"]) & np.isfinite(df["shift"])
        candidate_categories = ["1_DE_and_shift", "2_context_only"]
        candidates = df[finite & df["category"].isin(candidate_categories)].copy()
        x_span = (
            float(candidates["mean_expr"].max() - candidates["mean_expr"].min()) or 1.0
        )
        y_span = float(candidates["shift"].max() - candidates["shift"].min()) or 1.0
        candidates["_x_plot"] = (
            candidates["mean_expr"] - candidates["mean_expr"].min()
        ) / x_span
        candidates["_y_plot"] = (
            candidates["shift"] - candidates["shift"].min()
        ) / y_span
        coords = candidates[["_x_plot", "_y_plot"]].to_numpy(dtype=float)
        if len(coords) > 1:
            delta = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((delta * delta).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            candidates["_isolation"] = dist.min(axis=1)
        else:
            candidates["_isolation"] = 1.0
        selected = []
        selected_coords = []
        quotas = {"1_DE_and_shift": 7, "2_context_only": 7}
        min_sep = 0.07
        for category, quota in quotas.items():
            sub = candidates[candidates["category"] == category].copy()
            sub["_shift_rank"] = sub["shift"].rank(pct=True)
            sub["_isolation_rank"] = sub["_isolation"].rank(pct=True)
            sub["_label_score"] = (
                0.65 * sub["_shift_rank"] + 0.35 * sub["_isolation_rank"]
            )
            for idx, r in sub.sort_values("_label_score", ascending=False).iterrows():
                pt = np.array([float(r["_x_plot"]), float(r["_y_plot"])])
                if selected_coords:
                    d = np.sqrt(((np.vstack(selected_coords) - pt) ** 2).sum(axis=1))
                    if float(d.min()) < min_sep:
                        continue
                selected.append(idx)
                selected_coords.append(pt)
                if sum((df.loc[i, "category"] == category for i in selected)) >= quota:
                    break
        df["label"] = ""
        df.loc[selected, "label"] = df.loc[selected, "gene"]
        story_mask = df["gene"].isin(STORY_GENES)
        df.loc[story_mask, "label"] = df.loc[story_mask, "gene"]
        return df.reset_index(drop=True)

    umap_avg_xy = build_umap_avg(model_file)
    umap_avg_values = build_umap_avg_values(model_file, labels_file)
    umap_avg_meta = build_umap_avg_meta(labels_file)
    umap_shared_xy = build_umap_shared(model_file)
    umap_shared_values = build_umap_shared_values(model_file)
    umap_shared_meta = {}
    crc_spatial_payload = build_spatial_payload(crc_file, model_file)
    nat_spatial_payload = build_spatial_payload(nat_file, model_file)
    crc_spatial_meta = build_spatial_meta(crc_file, model_file)
    nat_spatial_meta = build_spatial_meta(nat_file, model_file)
    fig5_analysis = build_fig5_analysis(model_file, labels_file, all_sample_files)
    log2fc_df = fig5_analysis["log2fc_df"]
    shift_df = add_shift_labels(fig5_analysis["shift_df"])
    neighbor_overlap_df = fig5_analysis["neighbor_overlap_df"]
    least_neighbor_overlap_df = fig5_analysis["least_neighbor_overlap_df"]
    emp1_neighbors_df = fig5_analysis["emp1_neighbors_df"]
    return {
        "umap_avg_xy": umap_avg_xy,
        "umap_avg_values": umap_avg_values,
        "umap_avg_meta": umap_avg_meta,
        "umap_shared_xy": umap_shared_xy,
        "umap_shared_values": umap_shared_values,
        "umap_shared_meta": umap_shared_meta,
        "crc_spatial_payload": crc_spatial_payload,
        "nat_spatial_payload": nat_spatial_payload,
        "crc_spatial_meta": crc_spatial_meta,
        "nat_spatial_meta": nat_spatial_meta,
        "log2fc_df": log2fc_df,
        "shift_df": shift_df,
        "neighbor_overlap_df": neighbor_overlap_df,
        "least_neighbor_overlap_df": least_neighbor_overlap_df,
        "emp1_neighbors_df": emp1_neighbors_df,
    }


def panel_umap_avg(umap_avg_xy, umap_avg_values, umap_avg_meta):
    return {
        "x": umap_avg_xy[:, 0],
        "y": umap_avg_xy[:, 1],
        "values": umap_avg_values,
        "type": "categorical",
        "_meta": umap_avg_meta,
    }


def panel_umap_shared(umap_shared_xy, umap_shared_values, umap_shared_meta):
    return {
        "x": umap_shared_xy[:, 0],
        "y": umap_shared_xy[:, 1],
        "values": umap_shared_values,
        "type": "categorical",
        "_meta": umap_shared_meta,
    }


def panel_bar_log2fc(log2fc_df):
    items = [
        {"label": str(int(r["cluster"])), "value": float(r["log2_crc_over_nat"])}
        for (_, r) in log2fc_df.iterrows()
    ]
    bars = [{"name": "all", "items": items}]
    return {"bars": bars}


def panel_scatter_shift(shift_df):
    CATEGORY_LABELS = {
        "1_DE_and_shift": "Expr + shift",
        "2_context_only": "Shift only",
        "3_expr_only": "Expr only",
        "low": "Other",
    }
    STORY_LABEL_OFFSETS = {
        "DAB2": (10, 0, "start"),
        "CEACAM6": (10, 10, "start"),
        "EMP1": (10, 10, "start"),
        "CEACAM5": (10, 0, "start"),
        "LEFTY1": (10, 20, "start"),
        "CPNE1": (10, 30, "start"),
        "SLC7A5": (10, 40, "start"),
        "DHCR24": (10, 50, "start"),
        "AQP1": (10, 60, "start"),
        "TFF3": (10, 20, "start"),
    }
    points = []
    for _, r in shift_df.iterrows():
        category = CATEGORY_LABELS.get(str(r["category"]), str(r["category"]))
        pt = {"x": float(r["mean_expr"]), "y": float(r["shift"]), "category": category}
        label = str(r.get("label", ""))
        if label:
            pt["label"] = label
            if label in STORY_LABEL_OFFSETS:
                (pt["labelDx"], pt["labelDy"], pt["labelAnchor"]) = STORY_LABEL_OFFSETS[
                    label
                ]
        points.append(pt)
    categories = [
        {"name": "Expr + shift", "color": "#c94343"},
        {"name": "Shift only", "color": "#3891b8"},
        {"name": "Expr only", "color": "#b0b0b0"},
        {"name": "Other", "color": "#dcdcdc"},
    ]
    _meta = {"xlabel": "Mean log-expression", "ylabel": "CRC–NAT cosine distance"}
    return {"points": points, "categories": categories, "_meta": _meta}


def panel_bar_neighbor_replacement(neighbor_overlap_df, least_neighbor_overlap_df):
    high_bars = [
        {"label": str(r["gene"]), "value": float(r["replaced_n"])}
        for (_, r) in neighbor_overlap_df.sort_values(
            "replaced_n", ascending=False
        ).iterrows()
    ]
    low_bars = [
        {"label": str(r["gene"]), "value": float(r["replaced_n"])}
        for (_, r) in least_neighbor_overlap_df.sort_values(
            "replaced_n", ascending=False
        ).iterrows()
    ]
    bars = high_bars + [{"label": " ", "value": 0.0, "color": "transparent"}] + low_bars
    return {"bars": bars}


def panel_bar_emp1_neighbors(emp1_neighbors_df):
    crc = emp1_neighbors_df[emp1_neighbors_df["condition"] == "CRC"].sort_values(
        "similarity", ascending=False
    )
    nat = emp1_neighbors_df[emp1_neighbors_df["condition"] == "NAT"].sort_values(
        "similarity", ascending=False
    )
    crc_bars = [
        {"label": str(r["gene"]), "value": float(r["similarity"]), "color": "#5f8f6b"}
        for (_, r) in crc.iterrows()
    ]
    nat_bars = [
        {"label": str(r["gene"]), "value": float(r["similarity"]), "color": "#9b6676"}
        for (_, r) in nat.iterrows()
    ]
    bars = crc_bars + [{"label": " ", "value": 0.0, "color": "transparent"}] + nat_bars
    return {"bars": bars}


def panel_spatial_p2crc_c7(crc_spatial_payload, crc_spatial_meta):
    return {
        "x": crc_spatial_payload["x"],
        "y": crc_spatial_payload["y"],
        "values": crc_spatial_payload["cluster_7"],
        "type": "continuous",
        "_meta": crc_spatial_meta["cluster_7"],
    }


def panel_spatial_p5nat_c7(nat_spatial_payload, nat_spatial_meta):
    return {
        "x": nat_spatial_payload["x"],
        "y": nat_spatial_payload["y"],
        "values": nat_spatial_payload["cluster_7"],
        "type": "continuous",
        "_meta": nat_spatial_meta["cluster_7"],
    }


def panel_spatial_p2crc_c15(crc_spatial_payload, crc_spatial_meta):
    return {
        "x": crc_spatial_payload["x"],
        "y": crc_spatial_payload["y"],
        "values": crc_spatial_payload["cluster_15"],
        "type": "continuous",
        "_meta": crc_spatial_meta["cluster_15"],
    }


def panel_spatial_p5nat_c15(nat_spatial_payload, nat_spatial_meta):
    return {
        "x": nat_spatial_payload["x"],
        "y": nat_spatial_payload["y"],
        "values": nat_spatial_payload["cluster_15"],
        "type": "continuous",
        "_meta": nat_spatial_meta["cluster_15"],
    }


def panel_spatial_p2crc_c4(crc_spatial_payload, crc_spatial_meta):
    return {
        "x": crc_spatial_payload["x"],
        "y": crc_spatial_payload["y"],
        "values": crc_spatial_payload["cluster_4"],
        "type": "continuous",
        "_meta": crc_spatial_meta["cluster_4"],
    }


def panel_spatial_p5nat_c4(nat_spatial_payload, nat_spatial_meta):
    return {
        "x": nat_spatial_payload["x"],
        "y": nat_spatial_payload["y"],
        "values": nat_spatial_payload["cluster_4"],
        "type": "continuous",
        "_meta": nat_spatial_meta["cluster_4"],
    }


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_umap_avg(
            umap_avg_xy=results["umap_avg_xy"],
            umap_avg_values=results["umap_avg_values"],
            umap_avg_meta=results["umap_avg_meta"],
        ),
        "umap",
        output_dir / "umap-avg",
        "Gene modules (avg UMAP)",
        {},
    )
    save_panel(
        panel_umap_shared(
            umap_shared_xy=results["umap_shared_xy"],
            umap_shared_values=results["umap_shared_values"],
            umap_shared_meta=results["umap_shared_meta"],
        ),
        "umap",
        output_dir / "umap-shared",
        "CRC vs NAT (shared UMAP)",
        {},
    )
    save_panel(
        panel_bar_log2fc(log2fc_df=results["log2fc_df"]),
        "divergent",
        output_dir / "bar-log2fc",
        "Module enrichment: log2(CRC/NAT)",
        {"xlabel": "", "ylabel": ""},
    )
    save_panel(
        panel_scatter_shift(shift_df=results["shift_df"]),
        "scatter",
        output_dir / "scatter-shift",
        "Per-gene CRC–NAT embedding shift",
        {
            "xlabel": "Mean log-expression",
            "ylabel": "CRC–NAT cosine distance",
            "ymax": 0.45,
        },
    )
    save_panel(
        panel_bar_neighbor_replacement(
            neighbor_overlap_df=results["neighbor_overlap_df"],
            least_neighbor_overlap_df=results["least_neighbor_overlap_df"],
        ),
        "bar",
        output_dir / "bar-neighbor-replacement",
        "Neighborhood replacement: high and low",
        {"ylabel": "neighbors changed", "ymax": 25},
    )
    save_panel(
        panel_bar_emp1_neighbors(emp1_neighbors_df=results["emp1_neighbors_df"]),
        "bar",
        output_dir / "bar-emp1-neighbors",
        "EMP1 nearest genes by condition",
        {"ylabel": "cosine similarity", "ymax": 1, "ymin": 0},
    )
    save_panel(
        panel_spatial_p2crc_c7(
            crc_spatial_payload=results["crc_spatial_payload"],
            crc_spatial_meta=results["crc_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p2crc-c7",
        "P2-CRC (c7)",
        {},
    )
    save_panel(
        panel_spatial_p5nat_c7(
            nat_spatial_payload=results["nat_spatial_payload"],
            nat_spatial_meta=results["nat_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p5nat-c7",
        "P5-NAT (c7)",
        {},
    )
    save_panel(
        panel_spatial_p2crc_c15(
            crc_spatial_payload=results["crc_spatial_payload"],
            crc_spatial_meta=results["crc_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p2crc-c15",
        "P2-CRC (c15)",
        {},
    )
    save_panel(
        panel_spatial_p5nat_c15(
            nat_spatial_payload=results["nat_spatial_payload"],
            nat_spatial_meta=results["nat_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p5nat-c15",
        "P5-NAT (c15)",
        {},
    )
    save_panel(
        panel_spatial_p2crc_c4(
            crc_spatial_payload=results["crc_spatial_payload"],
            crc_spatial_meta=results["crc_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p2crc-c4",
        "P2-CRC (c4)",
        {},
    )
    save_panel(
        panel_spatial_p5nat_c4(
            nat_spatial_payload=results["nat_spatial_payload"],
            nat_spatial_meta=results["nat_spatial_meta"],
        ),
        "spatial",
        output_dir / "spatial-p5nat-c4",
        "P5-NAT (c4)",
        {},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "fig5")
