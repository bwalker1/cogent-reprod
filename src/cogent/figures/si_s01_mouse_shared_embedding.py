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

    def normalized_rows(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-09)

    def embedding_component(model):
        emb = np.asarray(model.get_embeddings(), dtype=np.float32)
        half = emb.shape[-1] // 2
        return emb[..., :half]

    def average_embedding(model):
        return embedding_component(model).mean(axis=0)

    def label_info(model, resolution):
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        size_map = dict(zip(df["cluster"].astype(int), df["n_genes"].astype(int)))
        return (labs, label_map, size_map)

    def build_shared_umap_payload():
        model = loaded_model
        coords = np.asarray(model._umap_coords["average"], dtype=np.float32)
        (labs, label_map, size_map) = label_info(model, resolution)
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

    def build_representative_genes_df():
        model = loaded_model
        avg = average_embedding(model)
        norm = normalized_rows(avg)
        (labs, label_map, size_map) = label_info(model, resolution)
        genes = np.asarray(model._gene_names, dtype=str)
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
            if len(idx) == 0:
                continue
            centroid = avg[idx].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-09)
            sims = norm[idx] @ centroid
            order = idx[np.argsort(-sims)[:6]]
            for rank, gene_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "cluster": int(cluster),
                        "label": label_map.get(int(cluster), "unlabeled"),
                        "n_genes": int(size_map.get(int(cluster), len(idx))),
                        "rank": int(rank),
                        "gene": str(genes[gene_idx]),
                        "centroid_similarity": float(
                            sims[np.where(idx == gene_idx)[0][0]]
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def build_within_between_similarity_df():
        model = loaded_model
        avg = normalized_rows(average_embedding(model))
        (labs, label_map, size_map) = label_info(model, resolution)
        rng = np.random.default_rng(7)
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
            if len(idx) < 2:
                continue
            n = min(250, len(idx) * 4)
            a = rng.choice(idx, size=n, replace=True)
            b = rng.choice(idx, size=n, replace=True)
            keep = a != b
            for value in np.sum(avg[a[keep]] * avg[b[keep]], axis=1):
                rows.append(
                    {
                        "comparison": "within",
                        "module": f"c{int(cluster)}",
                        "similarity": float(value),
                    }
                )
        valid = np.arange(len(labs))
        n_between = min(2500, len(valid) * 2)
        a = rng.choice(valid, size=n_between, replace=True)
        b = rng.choice(valid, size=n_between, replace=True)
        keep = labs[a] != labs[b]
        for value in np.sum(avg[a[keep]] * avg[b[keep]], axis=1):
            rows.append(
                {
                    "comparison": "between",
                    "module": "different",
                    "similarity": float(value),
                }
            )
        return pd.DataFrame(rows)

    def build_module_compactness_df():
        model = loaded_model
        avg = normalized_rows(average_embedding(model))
        (labs, label_map, size_map) = label_info(model, resolution)
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
            n = len(idx)
            if n < 2:
                mean_similarity = 0.0
            else:
                vec = avg[idx]
                similarity_sum = float(np.sum(vec @ vec.T) - n)
                mean_similarity = similarity_sum / (n * (n - 1))
            rows.append(
                {
                    "cluster": int(cluster),
                    "label": label_map.get(int(cluster), "unlabeled"),
                    "n_genes": int(size_map.get(int(cluster), n)),
                    "mean_within_similarity": float(mean_similarity),
                }
            )
        return pd.DataFrame(rows)

    shared_umap_payload = build_shared_umap_payload()
    module_size_df = (
        load_labels().sort_values("n_genes", ascending=False).reset_index(drop=True)
    )
    representative_genes_df = build_representative_genes_df()
    within_between_similarity_df = build_within_between_similarity_df()
    module_compactness_df = build_module_compactness_df()
    return {
        "shared_umap_payload": shared_umap_payload,
        "module_size_df": module_size_df,
        "representative_genes_df": representative_genes_df,
        "within_between_similarity_df": within_between_similarity_df,
        "module_compactness_df": module_compactness_df,
    }


def panel_s1_module_sizes(module_size_df):
    df = module_size_df.sort_values("n_genes", ascending=False)
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["n_genes"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s1_similarity_box(within_between_similarity_df):
    groups = [
        {"label": str(name), "values": sub["similarity"].astype(float).tolist()}
        for (name, sub) in within_between_similarity_df.groupby(
            "comparison", sort=False
        )
    ]
    return {"groups": groups}


def panel_s1_compactness_scatter(module_compactness_df):
    df = module_compactness_df.copy()
    points = [
        {
            "x": float(r["n_genes"]),
            "y": float(r["mean_within_similarity"]),
            "category": "modules",
            "label": f"c{int(r['cluster'])}",
        }
        for (_, r) in df.iterrows()
    ]
    categories = [{"name": "modules", "color": "#777777"}]
    return {"points": points, "categories": categories}


def panel_s1_representative_bars(representative_genes_df):
    df = representative_genes_df[representative_genes_df["rank"] == 1].copy()
    df = df.sort_values("centroid_similarity", ascending=True)
    bars = [
        {
            "label": f"c{int(r['cluster'])}: {r['gene']}",
            "value": float(r["centroid_similarity"]),
        }
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        results["shared_umap_payload"],
        "umap",
        output_dir / "s1-shared-embedding",
        "Shared embedding modules",
        {},
    )
    save_panel(
        panel_s1_module_sizes(module_size_df=results["module_size_df"]),
        "bar",
        output_dir / "s1-module-sizes",
        "Module sizes",
        {"ylabel": "genes"},
    )
    save_panel(
        panel_s1_similarity_box(
            within_between_similarity_df=results["within_between_similarity_df"]
        ),
        "boxplot",
        output_dir / "s1-similarity-box",
        "Intra- vs inter-module similarity",
        {"ylabel": "cosine similarity"},
    )
    save_panel(
        panel_s1_compactness_scatter(
            module_compactness_df=results["module_compactness_df"]
        ),
        "scatter",
        output_dir / "s1-compactness-scatter",
        "Module compactness",
        {"xlabel": "genes", "ylabel": "mean within-module similarity"},
    )
    save_panel(
        panel_s1_representative_bars(
            representative_genes_df=results["representative_genes_df"]
        ),
        "bar",
        output_dir / "s1-representative-bars",
        "Top representative genes",
        {"ylabel": "centroid similarity"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s01_mouse_shared_embedding")
