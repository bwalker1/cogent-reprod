from pathlib import Path
from cogent import Cogent
import numpy as np
import pandas as pd
from umap import UMAP
from cogent.figures.plotting import save_panel


def calculate(data_dir, resolution=0.5):
    data_dir = Path(data_dir)
    model_file = data_dir / "visium_crc/models/cogent/model.pt"
    labels_file = data_dir / "visium_crc/models/cogent/cluster_labels_res0.5.csv"

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

    def shared_payload(model):
        coords = np.asarray(model._umap_coords["average"], dtype=np.float32)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
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

    def condition_payloads(model):
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
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        values = [f"c{int(c)}: {label_map.get(int(c), 'unlabeled')}" for c in labs]
        return {
            "CRC": {
                "x": coords[0, :, 0],
                "y": coords[0, :, 1],
                "values": values,
                "type": "categorical",
            },
            "NAT": {
                "x": coords[1, :, 0],
                "y": coords[1, :, 1],
                "values": values,
                "type": "categorical",
            },
        }

    def centroid_shift(model):
        emb = embedding_component(model)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        rows = []
        for cluster in sorted(set(labs.astype(int))):
            idx = np.where(labs == cluster)[0]
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

    def displacement_df(model, n_arrows=180):
        payload = condition_payloads(model)
        shift = model_shift(model)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        genes = np.asarray(model._gene_names, dtype=str)
        order = np.argsort(-shift)[:n_arrows]
        return pd.DataFrame(
            {
                "gene": genes[order],
                "cluster": labs[order].astype(int),
                "x": payload["NAT"]["x"][order],
                "y": payload["NAT"]["y"][order],
                "u": payload["CRC"]["x"][order] - payload["NAT"]["x"][order],
                "v": payload["CRC"]["y"][order] - payload["NAT"]["y"][order],
                "shift": shift[order],
            }
        )

    model = load_model()
    HIGHLIGHT_GENES = [
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

    def build_highlight_gene_df():
        shift = model_shift(model)
        labs = np.asarray(
            model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        coords = np.asarray(model._umap_coords["average"], dtype=np.float32)
        genes = list(np.asarray(model._gene_names, dtype=str))
        rows = []
        for gene in HIGHLIGHT_GENES:
            if gene not in genes:
                continue
            idx = genes.index(gene)
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

    umap_avg_payload = shared_payload(model)
    condition_umap_payloads = condition_payloads(model)
    crc_nat_displacement_df = displacement_df(model)
    module_centroid_shift_df = centroid_shift(model)
    highlight_gene_df = build_highlight_gene_df()
    return {
        "umap_avg_payload": umap_avg_payload,
        "condition_umap_payloads": condition_umap_payloads,
        "crc_nat_displacement_df": crc_nat_displacement_df,
        "module_centroid_shift_df": module_centroid_shift_df,
        "highlight_gene_df": highlight_gene_df,
    }


def panel_s9_crc(condition_umap_payloads):
    return condition_umap_payloads["CRC"]


def panel_s9_nat(condition_umap_payloads):
    return condition_umap_payloads["NAT"]


def panel_s9_displacement(crc_nat_displacement_df):
    df = crc_nat_displacement_df
    x = df["x"].astype(float).tolist()
    y = df["y"].astype(float).tolist()
    u = df["u"].astype(float).tolist()
    v = df["v"].astype(float).tolist()
    magnitude = df["shift"].astype(float).tolist()
    return {"x": x, "y": y, "u": u, "v": v, "magnitude": magnitude}


def panel_s9_centroid(module_centroid_shift_df):
    df = module_centroid_shift_df[
        module_centroid_shift_df["n_genes"] >= 10
    ].sort_values("shift", ascending=False)
    bars = [
        {"label": f"c{int(r['cluster'])}", "value": float(r["shift"])}
        for (_, r) in df.iterrows()
    ]
    return {"bars": bars}


def panel_s9_highlight(highlight_gene_df):
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


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        results["umap_avg_payload"],
        "umap",
        output_dir / "s9-shared",
        "Shared CRC/NAT structure",
        {},
    )
    save_panel(
        panel_s9_crc(condition_umap_payloads=results["condition_umap_payloads"]),
        "umap",
        output_dir / "s9-crc",
        "CRC-specific embedding",
        {},
    )
    save_panel(
        panel_s9_nat(condition_umap_payloads=results["condition_umap_payloads"]),
        "umap",
        output_dir / "s9-nat",
        "NAT-specific embedding",
        {},
    )
    save_panel(
        panel_s9_displacement(
            crc_nat_displacement_df=results["crc_nat_displacement_df"]
        ),
        "quiver",
        output_dir / "s9-displacement",
        "CRC-NAT displacement",
        {},
    )
    save_panel(
        panel_s9_centroid(module_centroid_shift_df=results["module_centroid_shift_df"]),
        "bar",
        output_dir / "s9-centroid",
        "Module centroid shift",
        {"ylabel": "cosine distance"},
    )
    save_panel(
        panel_s9_highlight(highlight_gene_df=results["highlight_gene_df"]),
        "scatter",
        output_dir / "s9-highlight",
        "Highlighted epithelial genes",
        {"xlabel": "CRC-NAT shift", "ylabel": "module"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s09_crc_nat_condition_embeddings")
