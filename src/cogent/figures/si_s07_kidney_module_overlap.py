from pathlib import Path
from cogent import Cogent
import numpy as np
import pandas as pd
from cogent.figures.plotting import save_panel


def calculate(data_dir, resolution=0.5):
    data_dir = Path(data_dir)
    int_model_file = data_dir / "kidney-carcinoma/models/cogent/model.pt"
    int_labels_file = (
        data_dir / "kidney-carcinoma/models/cogent/cluster_labels_res0.5.csv"
    )
    sc_model_file = data_dir / "kidney-carcinoma/models/cogent_sc/model.pt"

    def load_int_model():
        return Cogent.load(Path(int_model_file).parent, device="cpu")

    def load_sc_model():
        return Cogent.load(Path(sc_model_file).parent, device="cpu")

    def shared_payload(model, labels_path):
        coords = np.asarray(model._umap_coords["average"], dtype=np.float32)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = pd.read_csv(labels_path)
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        size_map = dict(zip(df["cluster"].astype(int), df["n_genes"].astype(int)))
        values = [f"c{int(c)}: {label_map.get(int(c), 'unlabeled')}" for c in labs]
        meta = {
            "cluster_sizes": {
                f"c{int(c)}: {label_map.get(int(c), 'unlabeled')}": int(
                    size_map.get(int(c), 0)
                )
                for c in sorted(set(labs.astype(int)))
            }
        }
        return {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "values": values,
            "type": "categorical",
            "_meta": meta,
        }

    def cluster_jaccard(int_model, sc_model, labels_path):
        li = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        ls = np.asarray(
            sc_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = pd.read_csv(labels_path)
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        rows = []
        for ca in sorted(set(li.astype(int))):
            ma = li == ca
            best = {
                "sc_cluster": -1,
                "jaccard": 0.0,
                "intersect": 0,
                "union": int(ma.sum()),
            }
            for cb in sorted(set(ls.astype(int))):
                mb = ls == cb
                inter = int((ma & mb).sum())
                union = int((ma | mb).sum())
                score = inter / union if union else 0.0
                if score > best["jaccard"]:
                    best = {
                        "sc_cluster": int(cb),
                        "jaccard": float(score),
                        "intersect": inter,
                        "union": union,
                    }
            rows.append(
                {
                    "cluster": int(ca),
                    "label": label_map.get(int(ca), "unlabeled"),
                    "n_genes": int(ma.sum()),
                    **best,
                }
            )
        return pd.DataFrame(rows)

    def overlap_matrix(int_model, sc_model, labels_path):
        li = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        ls = np.asarray(
            sc_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = pd.read_csv(labels_path)
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        rows = []
        for ca in sorted(set(li.astype(int))):
            for cb in sorted(set(ls.astype(int))):
                ma = li == ca
                mb = ls == cb
                inter = int((ma & mb).sum())
                union = int((ma | mb).sum())
                rows.append(
                    {
                        "int_cluster": int(ca),
                        "int_label": label_map.get(int(ca), "unlabeled"),
                        "sc_cluster": int(cb),
                        "jaccard": float(inter / union if union else 0.0),
                        "intersect": inter,
                    }
                )
        return pd.DataFrame(rows)

    int_model = load_int_model()
    sc_model = load_sc_model()

    def build_module_highlight_payloads():
        int_payload = shared_payload(int_model, int_labels_file)
        sc_payload = shared_payload(sc_model, int_labels_file)
        li = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        out = {}
        for cluster in [2, 15, 17]:
            selected = li == cluster
            vals = ["selected" if s else "other" for s in selected]
            out[f"int_{cluster}"] = {
                "x": int_payload["x"],
                "y": int_payload["y"],
                "values": vals,
                "type": "categorical",
            }
            out[f"sc_{cluster}"] = {
                "x": sc_payload["x"],
                "y": sc_payload["y"],
                "values": vals,
                "type": "categorical",
            }
        return out

    def overlap_examples(best_df):
        li = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        ls = np.asarray(
            sc_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        rows = []
        for _, r in best_df.iterrows():
            cluster = int(r["cluster"])
            sc_cluster = int(r["sc_cluster"])
            int_set = set(np.where(li == cluster)[0].tolist())
            sc_set = set(np.where(ls == sc_cluster)[0].tolist())
            rows.append(
                {
                    "cluster": cluster,
                    "label": str(r["label"]),
                    "retained": len(int_set & sc_set),
                    "lost_from_int": len(int_set - sc_set),
                    "gained_in_sc": len(sc_set - int_set),
                    "jaccard": float(r["jaccard"]),
                }
            )
        return pd.DataFrame(rows)

    overlap_matrix_df = overlap_matrix(int_model, sc_model, int_labels_file)
    best_jaccard_df = cluster_jaccard(int_model, sc_model, int_labels_file)
    module_highlight_payloads = build_module_highlight_payloads()
    overlap_gene_examples_df = overlap_examples(best_jaccard_df)
    return {
        "overlap_matrix_df": overlap_matrix_df,
        "best_jaccard_df": best_jaccard_df,
        "module_highlight_payloads": module_highlight_payloads,
        "overlap_gene_examples_df": overlap_gene_examples_df,
    }


def panel_s7_overlap_matrix(overlap_matrix_df):
    df = overlap_matrix_df
    int_clusters = sorted(df["int_cluster"].unique())
    sc_clusters = sorted(df["sc_cluster"].unique())
    rows = [f"int c{int(c)}" for c in int_clusters]
    columns = [f"sc c{int(c)}" for c in sc_clusters]
    values = []
    for ca in int_clusters:
        sub = df[df["int_cluster"] == ca].set_index("sc_cluster")
        values.append(
            [
                float(sub.loc[cb, "jaccard"]) if cb in sub.index else 0.0
                for cb in sc_clusters
            ]
        )
    return {"rows": rows, "columns": columns, "values": values}


def panel_s7_best_ranking(best_jaccard_df):
    df = best_jaccard_df[best_jaccard_df["n_genes"] >= 10].sort_values(
        "jaccard", ascending=False
    )
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["jaccard"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_module_highlight(module_highlight_payloads, key):
    return module_highlight_payloads[key]


def panel_s7_weak_modules(best_jaccard_df):
    df = (
        best_jaccard_df[best_jaccard_df["n_genes"] >= 10]
        .sort_values("jaccard", ascending=True)
        .head(8)
    )
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["jaccard"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s7_overlap_examples(overlap_gene_examples_df):
    df = overlap_gene_examples_df.sort_values("jaccard", ascending=False).head(10)
    rows = [f"c{int(r['cluster'])}" for (_, r) in df.iterrows()]
    columns = ["retained", "lost", "gained"]
    values = [
        [float(r["retained"]), float(r["lost_from_int"]), float(r["gained_in_sc"])]
        for (_, r) in df.iterrows()
    ]
    return {"rows": rows, "columns": columns, "values": values}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s7_overlap_matrix(overlap_matrix_df=results["overlap_matrix_df"]),
        "heatmap",
        output_dir / "s7-overlap-matrix",
        "Full module-overlap matrix",
        {},
    )
    save_panel(
        panel_s7_best_ranking(best_jaccard_df=results["best_jaccard_df"]),
        "bar",
        output_dir / "s7-best-ranking",
        "Best-match Jaccard",
        {"ylabel": "best Jaccard"},
    )
    save_panel(
        panel_module_highlight(
            module_highlight_payloads=results["module_highlight_payloads"], key="int_2"
        ),
        "umap",
        output_dir / "s7-int-vascular",
        "Integrated vascular module",
        {},
    )
    save_panel(
        panel_module_highlight(
            module_highlight_payloads=results["module_highlight_payloads"], key="sc_2"
        ),
        "umap",
        output_dir / "s7-sc-vascular",
        "sc-only vascular genes",
        {},
    )
    save_panel(
        panel_module_highlight(
            module_highlight_payloads=results["module_highlight_payloads"], key="int_15"
        ),
        "umap",
        output_dir / "s7-int-bcell",
        "Integrated B/plasma",
        {},
    )
    save_panel(
        panel_module_highlight(
            module_highlight_payloads=results["module_highlight_payloads"], key="sc_15"
        ),
        "umap",
        output_dir / "s7-sc-bcell",
        "sc-only B/plasma",
        {},
    )
    save_panel(
        panel_s7_weak_modules(best_jaccard_df=results["best_jaccard_df"]),
        "bar",
        output_dir / "s7-weak-modules",
        "Module Recovery",
        {"ylabel": "best Jaccard"},
    )
    save_panel(
        panel_s7_overlap_examples(
            overlap_gene_examples_df=results["overlap_gene_examples_df"]
        ),
        "heatmap",
        output_dir / "s7-overlap-examples",
        "Gene Overlap",
        {},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s07_kidney_module_overlap")
