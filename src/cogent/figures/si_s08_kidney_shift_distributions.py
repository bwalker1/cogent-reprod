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

    def load_labels():
        return pd.read_csv(int_labels_file)

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

    def build_shift_comparison_df(int_model, sc_model, symbols):
        a = model_shift(int_model)
        b = model_shift(sc_model)
        return pd.DataFrame(
            {"gene": symbols, "integrated_shift": a, "sconly_shift": b, "delta": a - b}
        )

    int_model = load_int_model()
    sc_model = load_sc_model()
    kidney_symbols = symbols_for(int_model)

    def build_shift_long(comp):
        rows = []
        for _, r in comp.iterrows():
            rows.append(
                {
                    "gene": str(r["gene"]),
                    "model": "integrated",
                    "shift": float(r["integrated_shift"]),
                }
            )
            rows.append(
                {
                    "gene": str(r["gene"]),
                    "model": "sconly",
                    "shift": float(r["sconly_shift"]),
                }
            )
        return pd.DataFrame(rows)

    def build_tail_fraction(comp):
        thresholds = np.linspace(
            0.02,
            float(max(comp["integrated_shift"].max(), comp["sconly_shift"].max())),
            30,
        )
        rows = []
        for t in thresholds:
            rows.append(
                {
                    "threshold": float(t),
                    "model": "integrated",
                    "fraction": float((comp["integrated_shift"] >= t).mean()),
                }
            )
            rows.append(
                {
                    "threshold": float(t),
                    "model": "sconly",
                    "fraction": float((comp["sconly_shift"] >= t).mean()),
                }
            )
        return pd.DataFrame(rows)

    def build_module_stratified(comp):
        labs = np.asarray(
            int_model._leiden_results[resolution]["labels"]["average"], dtype=int
        )
        df = load_labels()
        label_map = dict(zip(df["cluster"].astype(int), df["label"].astype(str)))
        out = comp.copy()
        out["cluster"] = labs
        out["label"] = [label_map.get(int(c), "unlabeled") for c in labs]
        return out

    def build_top_overlap(comp):
        rows = []
        ri = np.argsort(-comp["integrated_shift"].to_numpy())
        rs = np.argsort(-comp["sconly_shift"].to_numpy())
        for k in [25, 50, 100, 200, 500]:
            shared = len(set(ri[:k].tolist()) & set(rs[:k].tolist()))
            rows.append({"k": int(k), "fraction": float(shared / k)})
        return pd.DataFrame(rows)

    shift_comparison_df = build_shift_comparison_df(int_model, sc_model, kidney_symbols)
    shift_long_df = build_shift_long(shift_comparison_df)
    tail_fraction_df = build_tail_fraction(shift_comparison_df)
    integrated_specific_shift_df = (
        shift_comparison_df.sort_values("delta", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    module_stratified_shift_df = build_module_stratified(shift_comparison_df)
    top_shift_overlap_df = build_top_overlap(shift_comparison_df)
    return {
        "shift_comparison_df": shift_comparison_df,
        "shift_long_df": shift_long_df,
        "tail_fraction_df": tail_fraction_df,
        "integrated_specific_shift_df": integrated_specific_shift_df,
        "module_stratified_shift_df": module_stratified_shift_df,
        "top_shift_overlap_df": top_shift_overlap_df,
    }


def panel_s8_shift_kde(shift_long_df):
    groups = [
        {"label": str(model), "values": sub["shift"].astype(float).tolist()}
        for (model, sub) in shift_long_df.groupby("model", sort=False)
    ]
    return {"groups": groups}


def panel_s8_tail(tail_fraction_df):
    series = []
    for model, sub in tail_fraction_df.groupby("model", sort=False):
        points = [
            {"x": float(r["threshold"]), "y": float(r["fraction"])}
            for (_, r) in sub.sort_values("threshold").iterrows()
        ]
        series.append({"name": str(model), "points": points})
    return {"series": series}


def panel_s8_shift_scatter(shift_comparison_df):
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


def panel_s8_integrated_specific(integrated_specific_shift_df):
    bars = [
        {"label": str(r["gene"]), "value": float(r["delta"])}
        for (_, r) in integrated_specific_shift_df.head(12).iterrows()
    ]
    return {"bars": bars}


def panel_s8_module_box(module_stratified_shift_df):
    top = (
        module_stratified_shift_df.groupby("cluster")["integrated_shift"]
        .median()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    groups = []
    for cluster in top:
        sub = module_stratified_shift_df[
            module_stratified_shift_df["cluster"] == cluster
        ]
        groups.append(
            {
                "label": f"c{int(cluster)}",
                "values": sub["integrated_shift"].astype(float).tolist(),
            }
        )
    return {"groups": groups}


def panel_s8_top_overlap(top_shift_overlap_df):
    points = [
        {"x": float(r["k"]), "y": float(r["fraction"])}
        for (_, r) in top_shift_overlap_df.iterrows()
    ]
    series = [{"name": "shared top-k", "points": points}]
    return {"series": series}


def run(data_dir, output_dir):
    import pickle

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(data_dir, resolution=0.5)
    with (output_dir / "analysis_1.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=5)
    save_panel(
        panel_s8_shift_kde(shift_long_df=results["shift_long_df"]),
        "kde",
        output_dir / "s8-shift-kde",
        "Shift distributions",
        {"xlabel": "condition shift"},
    )
    save_panel(
        panel_s8_tail(tail_fraction_df=results["tail_fraction_df"]),
        "line",
        output_dir / "s8-tail",
        "Large-shift tail fraction",
        {"xlabel": "shift threshold", "ylabel": "fraction"},
    )
    save_panel(
        panel_s8_shift_scatter(shift_comparison_df=results["shift_comparison_df"]),
        "scatter",
        output_dir / "s8-shift-scatter",
        "Per-gene shift comparison",
        {"xlabel": "sc-only shift", "ylabel": "integrated shift"},
    )
    save_panel(
        panel_s8_integrated_specific(
            integrated_specific_shift_df=results["integrated_specific_shift_df"]
        ),
        "bar",
        output_dir / "s8-integrated-specific",
        "Integrated-specific shifts",
        {"ylabel": "integrated - sc-only"},
    )
    save_panel(
        panel_s8_module_box(
            module_stratified_shift_df=results["module_stratified_shift_df"]
        ),
        "boxplot",
        output_dir / "s8-module-box",
        "Module-stratified integrated shifts",
        {"ylabel": "integrated shift"},
    )
    save_panel(
        panel_s8_top_overlap(top_shift_overlap_df=results["top_shift_overlap_df"]),
        "line",
        output_dir / "s8-top-overlap",
        "Top-shift overlap by cutoff",
        {"xlabel": "top-k genes", "ylabel": "overlap fraction"},
    )
    del results


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s08_kidney_shift_distributions")
