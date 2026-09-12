from pathlib import Path
from cogent import Cogent
import anndata as ad
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
    visium_file = data_dir / "kidney-carcinoma/raw/visium_interface.h5ad"

    def load_int_model():
        return Cogent.load(Path(int_model_file).parent, device="cpu")

    def load_sc_model():
        return Cogent.load(Path(sc_model_file).parent, device="cpu")

    def embedding_component(model):
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        return emb[..., :half]

    def normalized_rows(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-09)

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

    def build_shift_comparison_df(int_model, sc_model, symbols):
        a = model_shift(int_model)
        b = model_shift(sc_model)
        return pd.DataFrame(
            {"gene": symbols, "integrated_shift": a, "sconly_shift": b, "delta": a - b}
        )

    int_model = load_int_model()
    sc_model = load_sc_model()
    kidney_symbols = symbols_for(int_model)

    def build_pairwise_distance_df():
        ei = normalized_rows(embedding_component(int_model).mean(axis=0))
        es = normalized_rows(embedding_component(sc_model).mean(axis=0))
        rng = np.random.default_rng(11)
        n = ei.shape[0]
        a = rng.integers(0, n, size=2500)
        b = rng.integers(0, n, size=2500)
        keep = a != b
        a = a[keep]
        b = b[keep]
        return pd.DataFrame(
            {
                "integrated_distance": 1.0 - np.sum(ei[a] * ei[b], axis=1),
                "sconly_distance": 1.0 - np.sum(es[a] * es[b], axis=1),
            }
        )

    integrated_umap_payload = shared_payload(int_model, int_labels_file)
    sconly_umap_payload = shared_payload(sc_model, int_labels_file)
    pairwise_distance_df = build_pairwise_distance_df()
    best_jaccard_df = cluster_jaccard(int_model, sc_model, int_labels_file)
    shift_comparison_df = build_shift_comparison_df(int_model, sc_model, kidney_symbols)
    return {
        "integrated_umap_payload": integrated_umap_payload,
        "sconly_umap_payload": sconly_umap_payload,
        "pairwise_distance_df": pairwise_distance_df,
        "best_jaccard_df": best_jaccard_df,
        "shift_comparison_df": shift_comparison_df,
    }


def panel_s6_geometry(pairwise_distance_df):
    points = [
        {
            "x": float(r["sconly_distance"]),
            "y": float(r["integrated_distance"]),
            "category": "pairs",
        }
        for (_, r) in pairwise_distance_df.iterrows()
    ]
    categories = [{"name": "pairs", "color": "#777777"}]
    return {"points": points, "categories": categories}


def panel_s6_jaccard(best_jaccard_df):
    df = best_jaccard_df[best_jaccard_df["n_genes"] >= 10].sort_values(
        "jaccard", ascending=False
    )
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["jaccard"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s6_shift(shift_comparison_df):
    highlight = {"GPR183", "SLC40A1", "GPNMB", "KLF2", "CXCL12", "SEMA4B"}
    points = []
    for _, r in shift_comparison_df.iterrows():
        gene = str(r["gene"])
        points.append(
            {
                "x": float(r["sconly_shift"]),
                "y": float(r["integrated_shift"]),
                "category": "highlight" if gene in highlight else "other",
                "label": gene if gene in highlight else "",
            }
        )
    categories = [
        {"name": "highlight", "color": "#d97706"},
        {"name": "other", "color": "#b8b8b8"},
    ]
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
        results["integrated_umap_payload"],
        "umap",
        output_dir / "s6-integrated-umap",
        "Integrated embedding",
        {},
    )
    save_panel(
        results["sconly_umap_payload"],
        "umap",
        output_dir / "s6-sconly-umap",
        "Single-cell-only embedding",
        {},
    )
    save_panel(
        panel_s6_geometry(pairwise_distance_df=results["pairwise_distance_df"]),
        "scatter",
        output_dir / "s6-geometry",
        "Pairwise geometry comparison",
        {"xlabel": "sc-only cosine distance", "ylabel": "integrated cosine distance"},
    )
    save_panel(
        panel_s6_jaccard(best_jaccard_df=results["best_jaccard_df"]),
        "bar",
        output_dir / "s6-jaccard",
        "Best module preservation",
        {"ylabel": "best Jaccard"},
    )
    save_panel(
        panel_s6_shift(shift_comparison_df=results["shift_comparison_df"]),
        "scatter",
        output_dir / "s6-shift",
        "Condition-shift comparison",
        {"xlabel": "sc-only shift", "ylabel": "integrated shift"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s06_kidney_integrated_vs_sconly")
