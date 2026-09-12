from pathlib import Path
from cogent import Cogent
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cogent.figures.plotting import save_panel


def calculate(data_dir, resolution=0.5, sample="P5_NAT", k=25):
    data_dir = Path(data_dir)
    model_file = data_dir / "visium_crc/models/cogent/model.pt"
    labels_file = data_dir / "visium_crc/models/cogent/cluster_labels_res0.5.csv"
    p1_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P1_square_008um.h5ad"
    )
    p2_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P2_square_008um.h5ad"
    )
    p5_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P5_square_008um.h5ad"
    )
    p3_nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P3_square_008um.h5ad"
    )
    p5_nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P5_square_008um.h5ad"
    )
    SAMPLE_FILES = [
        ("P1_CRC", "CRC", p1_crc_file),
        ("P2_CRC", "CRC", p2_crc_file),
        ("P5_CRC", "CRC", p5_crc_file),
        ("P3_NAT", "NAT", p3_nat_file),
        ("P5_NAT", "NAT", p5_nat_file),
    ]

    def load_model():
        return Cogent.load(Path(model_file).parent, device="cpu")

    def load_labels():
        return pd.read_csv(labels_file)

    def embedding_component(model):
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        return emb[..., :half]

    def model_shift(model):
        emb = embedding_component(model)
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        return np.asarray(1.0 - np.sum(norm[0] * norm[1], axis=1), dtype=np.float32)

    def load_normalized_sample(path):
        a = ad.read_h5ad(path)
        if "in_tissue" in a.obs:
            a = a[a.obs["in_tissue"] == 1].copy()
        import scanpy as sc

        sc.pp.normalize_total(a, target_sum=10000.0)
        sc.pp.log1p(a)
        return a

    def dense_gene(a, gene):
        if gene not in a.var_names:
            return np.zeros(a.n_obs, dtype=np.float32)
        x = a[:, gene].X
        return (
            np.asarray(x.toarray() if sp.issparse(x) else x).ravel().astype(np.float32)
        )

    model = load_model()
    STROMAL_GENES = [
        "DAB2",
        "AQP1",
        "KLF2",
        "LUM",
        "NNMT",
        "SERPINF1",
        "PLVAP",
        "MYH11",
    ]

    def selected_sample_file():
        if sample == "P2_CRC":
            return p2_crc_file
        if sample == "P5_CRC":
            return p5_crc_file
        if sample == "P3_NAT":
            return p3_nat_file
        return p5_nat_file

    def build_shift_df():
        shift = model_shift(model)
        genes = np.asarray(model._gene_names, dtype=str)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        return pd.DataFrame(
            {
                "gene": genes,
                "shift": shift,
                "cluster": labs,
                "cluster_label": [label_map.get(int(c), "unlabeled") for c in labs],
                "is_stromal_highlight": [str(g) in STROMAL_GENES for g in genes],
            }
        )

    def expression_summary():
        genes = list(np.asarray(model._gene_names, dtype=str))
        rows = []
        for name, group, path in SAMPLE_FILES:
            a = load_normalized_sample(path)
            for gene in STROMAL_GENES:
                expr = dense_gene(a, gene)
                rows.append(
                    {
                        "sample": name,
                        "condition": group,
                        "gene": gene,
                        "mean_expr": float(np.mean(expr)),
                        "detected_fraction": float((expr > 0).mean()),
                    }
                )
        raw = pd.DataFrame(rows)
        out = []
        for gene, sub in raw.groupby("gene", sort=False):
            crc = sub[sub["condition"] == "CRC"]["mean_expr"].mean()
            nat = sub[sub["condition"] == "NAT"]["mean_expr"].mean()
            det_crc = sub[sub["condition"] == "CRC"]["detected_fraction"].mean()
            det_nat = sub[sub["condition"] == "NAT"]["detected_fraction"].mean()
            cluster = -1
            cluster_label = "missing"
            if gene in genes:
                idx = genes.index(gene)
                labs = np.asarray(
                    model._leiden_results[resolution]["labels"]["average"], dtype=int
                )
                label_map = dict(
                    zip(
                        load_labels()["cluster"].astype(int),
                        load_labels()["label"].astype(str),
                    )
                )
                cluster = int(labs[idx])
                cluster_label = label_map.get(cluster, "unlabeled")
            out.append(
                {
                    "gene": gene,
                    "cluster": cluster,
                    "cluster_label": cluster_label,
                    "crc_mean": float(crc),
                    "nat_mean": float(nat),
                    "log2_crc_over_nat": float(
                        np.log2((crc + 0.0001) / (nat + 0.0001))
                    ),
                    "det_crc": float(det_crc),
                    "det_nat": float(det_nat),
                }
            )
        return pd.DataFrame(out)

    def build_spatial_payload():
        a = load_normalized_sample(selected_sample_file())
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        out = {"x": coords[:, 0], "y": -coords[:, 1]}
        for gene in STROMAL_GENES:
            out[gene] = dense_gene(a, gene)
        return out

    def neighbor_context():
        genes = np.asarray(model._gene_names, dtype=str)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        label_map = dict(
            zip(
                load_labels()["cluster"].astype(int), load_labels()["label"].astype(str)
            )
        )
        emb = embedding_component(model)
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        rows = []
        for gene in STROMAL_GENES:
            if gene not in genes:
                continue
            idx = int(np.where(genes == gene)[0][0])
            for gi, condition in [(0, "CRC"), (1, "NAT")]:
                sims = norm[gi] @ norm[gi, idx]
                sims[idx] = -np.inf
                nbrs = np.argsort(-sims)[:25]
                counts = pd.Series(labs[nbrs]).value_counts()
                for cluster, count in counts.items():
                    rows.append(
                        {
                            "gene": gene,
                            "condition": condition,
                            "cluster": int(cluster),
                            "cluster_label": label_map.get(int(cluster), "unlabeled"),
                            "fraction": float(count / 25.0),
                        }
                    )
        return pd.DataFrame(rows)

    shift_df = build_shift_df()
    stromal_gene_summary_df = expression_summary()
    spatial_payload = build_spatial_payload()
    gene_condition_expression_df = expression_summary()
    neighbor_module_context_df = neighbor_context()

    def replacement_df_for(k_value):
        genes = np.asarray(model._gene_names, dtype=str)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        shift = model_shift(model)
        emb = embedding_component(model)
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        rows = []
        for idx, gene in enumerate(genes):
            crc_sims = norm[0] @ norm[0, idx]
            nat_sims = norm[1] @ norm[1, idx]
            crc_sims[idx] = -np.inf
            nat_sims[idx] = -np.inf
            crc = np.argsort(-crc_sims)[:k_value]
            nat = np.argsort(-nat_sims)[:k_value]
            shared = len(set(crc.tolist()) & set(nat.tolist()))
            union = len(set(crc.tolist()) | set(nat.tolist()))
            rows.append(
                {
                    "gene": str(gene),
                    "cluster": int(labs[idx]),
                    "shared_neighbors": int(shared),
                    "changed_neighbors": int(k_value - shared),
                    "jaccard": float(shared / union if union else 0.0),
                    "shift": float(shift[idx]),
                }
            )
        return pd.DataFrame(rows)

    def emp1_neighbors():
        genes = np.asarray(model._gene_names, dtype=str)
        if "EMP1" not in genes:
            return pd.DataFrame(
                columns=["condition", "rank", "gene", "similarity", "status"]
            )
        idx = int(np.where(genes == "EMP1")[0][0])
        emb = embedding_component(model)
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        sets = {}
        rows = []
        for gi, condition in [(0, "CRC"), (1, "NAT")]:
            sims = norm[gi] @ norm[gi, idx]
            sims[idx] = -np.inf
            nbrs = np.argsort(-sims)[:10]
            sets[condition] = set(nbrs.tolist())
            for rank, ni in enumerate(nbrs, start=1):
                rows.append(
                    {
                        "condition": condition,
                        "rank": int(rank),
                        "gene": str(genes[ni]),
                        "similarity": float(sims[ni]),
                        "neighbor_index": int(ni),
                    }
                )
        out = pd.DataFrame(rows)
        shared = sets.get("CRC", set()) & sets.get("NAT", set())
        out["status"] = [
            "shared" if int(i) in shared else str(cond)
            for (i, cond) in zip(out["neighbor_index"], out["condition"])
        ]
        return out.drop(columns=["neighbor_index"])

    neighbor_replacement_df = replacement_df_for(k)
    shift_vs_replacement_df = neighbor_replacement_df.copy()
    top_changed_neighbors_df = (
        neighbor_replacement_df.sort_values(
            ["changed_neighbors", "shift"], ascending=False
        )
        .head(15)
        .reset_index(drop=True)
    )
    stable_neighbor_controls_df = (
        neighbor_replacement_df.sort_values(
            ["changed_neighbors", "shift"], ascending=True
        )
        .head(15)
        .reset_index(drop=True)
    )
    emp1_neighbors_df = emp1_neighbors()
    return {
        "shift_df": shift_df,
        "stromal_gene_summary_df": stromal_gene_summary_df,
        "spatial_payload": spatial_payload,
        "gene_condition_expression_df": gene_condition_expression_df,
        "neighbor_module_context_df": neighbor_module_context_df,
        "neighbor_replacement_df": neighbor_replacement_df,
        "shift_vs_replacement_df": shift_vs_replacement_df,
        "top_changed_neighbors_df": top_changed_neighbors_df,
        "stable_neighbor_controls_df": stable_neighbor_controls_df,
        "emp1_neighbors_df": emp1_neighbors_df,
    }


def panel_s10_shift_context(shift_df):
    points = []
    for _, r in shift_df.iterrows():
        points.append(
            {
                "x": float(r["cluster"]),
                "y": float(r["shift"]),
                "category": "stromal" if bool(r["is_stromal_highlight"]) else "other",
                "label": str(r["gene"]) if bool(r["is_stromal_highlight"]) else "",
            }
        )
    categories = [
        {"name": "stromal", "color": "#d97706"},
        {"name": "other", "color": "#b8b8b8"},
    ]
    return {"points": points, "categories": categories}


def panel_s10_module_assignments(shift_df, stromal_gene_summary_df):
    df = stromal_gene_summary_df
    rows = df["gene"].astype(str).tolist()
    columns = ["module", "shift proxy", "log2 CRC/NAT"]
    shift_map = shift_df.set_index("gene")["shift"].to_dict()
    values = [
        [
            float(r["cluster"]),
            float(shift_map.get(str(r["gene"]), 0.0)),
            float(r["log2_crc_over_nat"]),
        ]
        for (_, r) in df.iterrows()
    ]
    return {"rows": rows, "columns": columns, "values": values}


def panel_spatial_gene(spatial_payload, gene):
    return {
        "x": spatial_payload["x"],
        "y": spatial_payload["y"],
        "values": spatial_payload[gene],
        "type": "continuous",
    }


def panel_s10_expression(gene_condition_expression_df):
    bars = []
    for _, r in gene_condition_expression_df.iterrows():
        bars.append({"label": str(r["gene"]), "value": float(r["log2_crc_over_nat"])})
    return {"bars": bars}


def panel_s10_neighbor_context(neighbor_module_context_df):
    df = neighbor_module_context_df
    keep_genes = ["DAB2", "AQP1", "KLF2", "LUM"]
    rows = [f"{gene} {cond}" for gene in keep_genes for cond in ["CRC", "NAT"]]
    clusters = sorted(df["cluster"].unique())[:10]
    columns = [f"c{int(c)}" for c in clusters]
    values = []
    for gene in keep_genes:
        for cond in ["CRC", "NAT"]:
            sub = df[(df["gene"] == gene) & (df["condition"] == cond)].set_index(
                "cluster"
            )
            values.append(
                [
                    float(sub.loc[c, "fraction"]) if c in sub.index else 0.0
                    for c in clusters
                ]
            )
    return {"rows": rows, "columns": columns, "values": values}


def panel_s11_method_summary(neighbor_replacement_df):
    bars = [
        {
            "label": "mean shared",
            "value": float(neighbor_replacement_df["shared_neighbors"].mean()),
        },
        {
            "label": "mean changed",
            "value": float(neighbor_replacement_df["changed_neighbors"].mean()),
        },
        {
            "label": "median changed",
            "value": float(neighbor_replacement_df["changed_neighbors"].median()),
        },
    ]
    return {"bars": bars}


def panel_s11_distribution(neighbor_replacement_df):
    groups = [
        {
            "label": "genes",
            "values": neighbor_replacement_df["changed_neighbors"]
            .astype(float)
            .tolist(),
        }
    ]
    return {"groups": groups}


def panel_s11_shift_scatter(shift_vs_replacement_df):
    points = [
        {
            "x": float(r["changed_neighbors"]),
            "y": float(r["shift"]),
            "category": "genes",
            "label": str(r["gene"])
            if str(r["gene"]) in {"EMP1", "DAB2", "CEACAM6", "CEACAM5"}
            else "",
        }
        for (_, r) in shift_vs_replacement_df.iterrows()
    ]
    categories = [{"name": "genes", "color": "#777777"}]
    return {"points": points, "categories": categories}


def panel_s11_top_changed(top_changed_neighbors_df):
    bars = [
        {"label": str(r["gene"]), "value": float(r["changed_neighbors"])}
        for (_, r) in top_changed_neighbors_df.head(12).iterrows()
    ]
    return {"bars": bars}


def panel_s11_stable_controls(stable_neighbor_controls_df):
    bars = [
        {"label": str(r["gene"]), "value": float(r["changed_neighbors"])}
        for (_, r) in stable_neighbor_controls_df.head(12).iterrows()
    ]
    return {"bars": bars}


def panel_s11_emp1(emp1_neighbors_df):
    df = emp1_neighbors_df.groupby("condition", sort=False).head(6)
    bars = [
        {
            "label": f"{r['condition']} {r['rank']}: {r['gene']}",
            "value": float(r["similarity"]),
        }
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def calculate_maps(data_dir, sample="P5_NAT"):
    data_dir = Path(data_dir)
    p2_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P2_square_008um.h5ad"
    )
    p5_crc_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Cancer_P5_square_008um.h5ad"
    )
    p3_nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P3_square_008um.h5ad"
    )
    p5_nat_file = (
        data_dir / "visium_crc/raw/Visium_HD_Human_Colon_Normal_P5_square_008um.h5ad"
    )

    def load_normalized_sample(path):
        a = ad.read_h5ad(path)
        if "in_tissue" in a.obs:
            a = a[a.obs["in_tissue"] == 1].copy()
        import scanpy as sc

        sc.pp.normalize_total(a, target_sum=10000.0)
        sc.pp.log1p(a)
        return a

    def dense_gene(a, gene):
        if gene not in a.var_names:
            return np.zeros(a.n_obs, dtype=np.float32)
        x = a[:, gene].X
        return (
            np.asarray(x.toarray() if sp.issparse(x) else x).ravel().astype(np.float32)
        )

    STROMAL_GENES = [
        "DAB2",
        "AQP1",
        "KLF2",
        "LUM",
        "NNMT",
        "SERPINF1",
        "PLVAP",
        "MYH11",
    ]

    def selected_sample_file():
        if sample == "P2_CRC":
            return p2_crc_file
        if sample == "P5_CRC":
            return p5_crc_file
        if sample == "P3_NAT":
            return p3_nat_file
        return p5_nat_file

    def build_spatial_payload():
        a = load_normalized_sample(selected_sample_file())
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        out = {"x": coords[:, 0], "y": -coords[:, 1]}
        for gene in STROMAL_GENES:
            out[gene] = dense_gene(a, gene)
        return out

    spatial_payload = build_spatial_payload()
    return {
        "spatial_payload": spatial_payload,
    }


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5, sample="P5_NAT", k=25)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s10_shift_context(shift_df=results["shift_df"]),
        "scatter",
        output_dir / "s10-shift-context",
        "Stromal gene shift",
        {"xlabel": "module", "ylabel": "CRC-NAT shift"},
    )
    save_panel(
        panel_s10_module_assignments(
            shift_df=results["shift_df"],
            stromal_gene_summary_df=results["stromal_gene_summary_df"],
        ),
        "heatmap",
        output_dir / "s10-module-assignments",
        "Stromal gene modules",
        {},
    )
    save_panel(
        panel_spatial_gene(spatial_payload=results["spatial_payload"], gene="DAB2"),
        "spatial",
        output_dir / "s10-nat-dab2",
        "NAT DAB2",
        {},
    )
    save_panel(
        panel_spatial_gene(spatial_payload=results["spatial_payload"], gene="AQP1"),
        "spatial",
        output_dir / "s10-nat-aqp1",
        "NAT AQP1",
        {},
    )
    save_panel(
        panel_s10_expression(
            gene_condition_expression_df=results["gene_condition_expression_df"]
        ),
        "bar",
        output_dir / "s10-expression",
        "CRC versus NAT expression",
        {"ylabel": "log2 CRC/NAT"},
    )
    save_panel(
        panel_s10_neighbor_context(
            neighbor_module_context_df=results["neighbor_module_context_df"]
        ),
        "heatmap",
        output_dir / "s10-neighbor-context",
        "Neighbor module context",
        {},
    )
    save_panel(
        panel_s11_method_summary(
            neighbor_replacement_df=results["neighbor_replacement_df"]
        ),
        "bar",
        output_dir / "s11-method-summary",
        "KNN replacement summary",
        {"ylabel": "neighbors"},
    )
    save_panel(
        panel_s11_distribution(
            neighbor_replacement_df=results["neighbor_replacement_df"]
        ),
        "histogram",
        output_dir / "s11-distribution",
        "Replacement distribution",
        {"xlabel": "changed neighbors"},
    )
    save_panel(
        panel_s11_shift_scatter(
            shift_vs_replacement_df=results["shift_vs_replacement_df"]
        ),
        "scatter",
        output_dir / "s11-shift-scatter",
        "Shift versus neighborhood change",
        {"xlabel": "changed neighbors", "ylabel": "CRC-NAT shift"},
    )
    save_panel(
        panel_s11_top_changed(
            top_changed_neighbors_df=results["top_changed_neighbors_df"]
        ),
        "bar",
        output_dir / "s11-top-changed",
        "Top changed neighborhoods",
        {"ylabel": "changed neighbors"},
    )
    save_panel(
        panel_s11_stable_controls(
            stable_neighbor_controls_df=results["stable_neighbor_controls_df"]
        ),
        "bar",
        output_dir / "s11-stable-controls",
        "Least changed neighborhoods",
        {"ylabel": "changed neighbors"},
    )
    save_panel(
        panel_s11_emp1(emp1_neighbors_df=results["emp1_neighbors_df"]),
        "bar",
        output_dir / "s11-emp1",
        "EMP1 local neighbors",
        {"ylabel": "cosine similarity"},
    )
    del results
    results = calculate_maps(data_dir, sample="P2_CRC")
    with (output_dir / "analysis_2.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_spatial_gene(spatial_payload=results["spatial_payload"], gene="DAB2"),
        "spatial",
        output_dir / "s10-crc-dab2",
        "CRC DAB2",
        {},
    )
    save_panel(
        panel_spatial_gene(spatial_payload=results["spatial_payload"], gene="AQP1"),
        "spatial",
        output_dir / "s10-crc-aqp1",
        "CRC AQP1",
        {},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s10_crc_nat_stromal_genes")
