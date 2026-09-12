from pathlib import Path
from cogent import Cogent
import numpy as np
import pandas as pd
from cogent.figures.plotting import save_panel


def calculate(data_dir, resolution=0.5):
    data_dir = Path(data_dir)
    model_file = data_dir / "mosta/models/cogent/model.pt"
    labels_file = data_dir / "mosta/models/cogent/cluster_labels_res0.5.csv"

    loaded_model = Cogent.load(Path(model_file).parent, device="cpu")

    def load_labels():
        return pd.read_csv(labels_file)

    def embedding_component(model):
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        return emb[..., :half]

    def label_info(model, resolution):
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        size_map = dict(zip(df["cluster"].astype(int), df["n_genes"].astype(int)))
        return (labs, label_map, size_map)

    STAGE_LABELS = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]

    def stage_indices(model):
        group_ids = [str(g) for g in model.group_ids or []]
        idxs = []
        for short in ["12", "13", "14", "15", "16"]:
            idxs.append(group_ids.index(short) if short in group_ids else len(idxs))
        return idxs

    def build_gene_shift_df():
        model = loaded_model
        emb = embedding_component(model)[stage_indices(model)]
        norm = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-09)
        genes = np.asarray(model._gene_names, dtype=str)
        (labs, label_map, size_map) = label_info(model, resolution)
        rows = []
        for i in range(len(STAGE_LABELS) - 1):
            shift = 1.0 - np.sum(norm[i] * norm[i + 1], axis=1)
            for gi, value in enumerate(shift):
                rows.append(
                    {
                        "gene": str(genes[gi]),
                        "cluster": int(labs[gi]),
                        "label": label_map.get(int(labs[gi]), "unlabeled"),
                        "transition": f"{STAGE_LABELS[i]}->{STAGE_LABELS[i + 1]}",
                        "shift": float(value),
                    }
                )
        return pd.DataFrame(rows)

    def build_cluster_shift_mat():
        model = loaded_model
        emb = embedding_component(model)[stage_indices(model)]
        (labs, label_map, size_map) = label_info(model, resolution)
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
            if len(idx) == 0:
                continue
            cent = emb[:, idx, :].mean(axis=1)
            cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-09)
            rec = {
                "cluster": int(cluster),
                "label": label_map.get(int(cluster), "unlabeled"),
                "n_genes": int(size_map.get(int(cluster), len(idx))),
            }
            vals = []
            for i in range(len(STAGE_LABELS) - 1):
                value = float(1.0 - np.dot(cent[i], cent[i + 1]))
                rec[f"{STAGE_LABELS[i]}->{STAGE_LABELS[i + 1]}"] = value
                vals.append(value)
            rec["mean_shift"] = float(np.mean(vals))
            rows.append(rec)
        return pd.DataFrame(rows)

    def build_aggregate_shift_df(cluster_df, gene_df):
        rows = []
        transition_cols = [c for c in cluster_df.columns if "->" in c]
        for col in transition_cols:
            weights = cluster_df["n_genes"].astype(float)
            weighted = float((cluster_df[col] * weights).sum() / weights.sum())
            sub = gene_df[gene_df["transition"] == col]
            rows.append(
                {
                    "transition": col,
                    "metric": "weighted module centroid",
                    "shift": weighted,
                }
            )
            rows.append(
                {
                    "transition": col,
                    "metric": "median gene",
                    "shift": float(sub["shift"].median()),
                }
            )
            rows.append(
                {
                    "transition": col,
                    "metric": "mean gene",
                    "shift": float(sub["shift"].mean()),
                }
            )
        return pd.DataFrame(rows)

    def build_top_early_modules(cluster_df):
        first = f"{STAGE_LABELS[0]}->{STAGE_LABELS[1]}"
        return (
            cluster_df.sort_values(first, ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

    gene_shift_df = build_gene_shift_df()
    cluster_shift_mat = build_cluster_shift_mat()
    aggregate_shift_df = build_aggregate_shift_df(cluster_shift_mat, gene_shift_df)
    top_early_modules_df = build_top_early_modules(cluster_shift_mat)
    return {
        "gene_shift_df": gene_shift_df,
        "cluster_shift_mat": cluster_shift_mat,
        "aggregate_shift_df": aggregate_shift_df,
        "top_early_modules_df": top_early_modules_df,
    }


def panel_s3_gene_shift_box(gene_shift_df):
    groups = [
        {"label": tr.replace("->", "\n"), "values": sub["shift"].astype(float).tolist()}
        for (tr, sub) in gene_shift_df.groupby("transition", sort=False)
    ]
    return {"groups": groups}


def panel_s3_module_shift_heatmap(cluster_shift_mat):
    df = cluster_shift_mat[cluster_shift_mat["n_genes"] >= 20].sort_values(
        "mean_shift", ascending=False
    )
    cols = [c for c in df.columns if "->" in c]
    rows = [f"c{int(r['cluster'])}" for (_, r) in df.iterrows()]
    columns = [c.replace("->", " to ") for c in cols]
    values = [[float(r[c]) for c in cols] for (_, r) in df.iterrows()]
    return {"rows": rows, "columns": columns, "values": values}


def panel_s3_aggregate_line(aggregate_shift_df):
    series = []
    for metric, sub in aggregate_shift_df.groupby("metric", sort=False):
        points = [
            {"x": i + 1, "y": float(r["shift"])}
            for (i, (_, r)) in enumerate(sub.iterrows())
        ]
        series.append({"name": str(metric), "points": points})
    return {"series": series}


def panel_s3_top_early_modules(top_early_modules_df):
    first = "E12.5->E13.5"
    df = top_early_modules_df
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r[first])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s3_top_shifting_genes(gene_shift_df, cluster_shift_mat):
    transitions = [c for c in cluster_shift_mat.columns if "->" in c]
    top_genes = []
    for tr in transitions:
        sub = (
            gene_shift_df[gene_shift_df["transition"] == tr]
            .sort_values("shift", ascending=False)
            .head(4)
        )
        top_genes.extend((str(g) for g in sub["gene"]))
    genes = list(dict.fromkeys(top_genes))
    rows = genes
    columns = [tr.replace("->", " to ") for tr in transitions]
    values = []
    by_gene = gene_shift_df.set_index(["gene", "transition"])
    for gene in genes:
        values.append(
            [
                float(by_gene.loc[(gene, tr), "shift"])
                if (gene, tr) in by_gene.index
                else 0.0
                for tr in transitions
            ]
        )
    return {"rows": rows, "columns": columns, "values": values}


def panel_s3_shift_size_scatter(cluster_shift_mat):
    df = cluster_shift_mat.copy()
    points = [
        {
            "x": float(r["n_genes"]),
            "y": float(r["mean_shift"]),
            "category": "modules",
            "label": f"c{int(r['cluster'])}",
        }
        for (_, r) in df.iterrows()
    ]
    categories = [{"name": "modules", "color": "#777777"}]
    return {"points": points, "categories": categories}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s3_gene_shift_box(gene_shift_df=results["gene_shift_df"]),
        "boxplot",
        output_dir / "s3-gene-shift-box",
        "Per-gene adjacent shifts",
        {"ylabel": "cosine distance"},
    )
    save_panel(
        panel_s3_module_shift_heatmap(cluster_shift_mat=results["cluster_shift_mat"]),
        "heatmap",
        output_dir / "s3-module-shift-heatmap",
        "Module centroid shifts",
        {},
    )
    save_panel(
        panel_s3_aggregate_line(aggregate_shift_df=results["aggregate_shift_df"]),
        "line",
        output_dir / "s3-aggregate-line",
        "Aggregate transition shift",
        {"xlabel": "transition index", "ylabel": "shift"},
    )
    save_panel(
        panel_s3_top_early_modules(
            top_early_modules_df=results["top_early_modules_df"]
        ),
        "bar",
        output_dir / "s3-top-early-modules",
        "Top E12.5 to E13.5 modules",
        {"ylabel": "cosine distance"},
    )
    save_panel(
        panel_s3_top_shifting_genes(
            gene_shift_df=results["gene_shift_df"],
            cluster_shift_mat=results["cluster_shift_mat"],
        ),
        "heatmap",
        output_dir / "s3-top-shifting-genes",
        "Top shifting genes",
        {},
    )
    save_panel(
        panel_s3_shift_size_scatter(cluster_shift_mat=results["cluster_shift_mat"]),
        "scatter",
        output_dir / "s3-shift-size-scatter",
        "Shift versus module size",
        {"xlabel": "genes", "ylabel": "mean centroid shift"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s04_mouse_developmental_shifts")
