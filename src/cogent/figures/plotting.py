"""Save figure panels and the numerical values used to draw them."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from cogent.figures.dual_channel import COLOR1, COLOR2, mix_colors, normalize_channels


def spatial_panel(xy, values, values2=None):
    """Use the manuscript's tissue orientation for single or paired signals."""
    data = {"x": xy[:, 0], "y": -xy[:, 1], "values": values, "type": "continuous"}
    if values2 is not None:
        data["values2"] = values2
    return data


def save_panel(data, kind, path, title, labels):
    path = Path(path)
    with path.with_suffix(".pkl").open("wb") as handle:
        pickle.dump(data, handle, protocol=5)
    fig, ax = plt.subplots(figsize=(6, 4))
    if kind in ("umap", "spatial"):
        x, y, values = (
            np.asarray(data["x"]),
            np.asarray(data["y"]),
            np.asarray(data["values"]),
        )
        if data.get("type") == "categorical":
            codes, categories = pd.factorize(values, sort=False)
            palette = plt.get_cmap("tab20", len(categories))
            ax.scatter(x, y, c=palette(codes), s=2, linewidths=0, rasterized=True)
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=palette(i),
                    label=name,
                    markersize=3,
                )
                for i, name in enumerate(categories)
            ]
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=5,
                frameon=False,
            )
        else:
            if "values2" in data:
                first, second = normalize_channels(values, data["values2"])
                color1, color2 = (
                    labels.get("color1", COLOR1),
                    labels.get("color2", COLOR2),
                )
                colors = mix_colors(first, second, color1, color2)
                visible = (values > 0) | (np.asarray(data["values2"]) > 0)
                order = np.flatnonzero(visible)
                order = order[
                    np.argsort(
                        (values + np.asarray(data["values2"]))[order], kind="stable"
                    )
                ]
                ax.set_facecolor("black")
                ax.scatter(
                    x[order],
                    y[order],
                    c=colors[order],
                    s=2,
                    linewidths=0,
                    rasterized=True,
                )
                names = labels.get("channel_labels", ["Channel 1", "Channel 2"])
                endpoints = mix_colors(
                    np.array([1, 0, 1]), np.array([0, 1, 1]), color1, color2
                )
                handles = [
                    plt.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        color=color,
                        label=name,
                        markersize=5,
                    )
                    for color, name in zip(endpoints, [*names, "Both"])
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    fontsize=7,
                    frameon=False,
                    title="Scaled signal",
                )
            else:
                plot = ax.scatter(
                    x, y, c=values, cmap="Blues", s=2, linewidths=0, rasterized=True
                )
                fig.colorbar(plot, ax=ax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    elif kind in ("heatmap", "clustermap"):
        values = np.asarray(data["values"], dtype=float)
        symmetric = data.get("symmetric", labels.get("symmetric", False))
        cmap = labels.get(
            "colorscale",
            "coolwarm" if symmetric or np.nanmin(values) < 0 else "viridis",
        )
        limits = {}
        if symmetric:
            finite = values[np.isfinite(values)]
            bound = float(np.max(np.abs(finite))) if finite.size else 1.0
            bound = bound or 1.0
            limits = dict(vmin=-bound, vmax=bound)
        fig.colorbar(
            ax.imshow(
                values,
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                rasterized=True,
                **limits,
            ),
            ax=ax,
        )
        if len(data["rows"]) < 60:
            ax.set_yticks(range(len(data["rows"])), data["rows"], fontsize=6)
        else:
            ax.set_yticks([])
        if len(data["columns"]) < 60:
            ax.set_xticks(
                range(len(data["columns"])), data["columns"], rotation=90, fontsize=6
            )
        else:
            ax.set_xticks([])
    elif kind in ("bar", "divergent"):
        bars = data["bars"]
        if bars and "items" in bars[0]:
            bars = [item for series in bars for item in series["items"]]
        table = pd.DataFrame(bars)
        colors = [
            b.get("color", "#4c78a8" if b["value"] >= 0 else "#e45756") for b in bars
        ]
        ax.bar(
            np.arange(len(table)),
            table["value"],
            color=["none" if c == "transparent" else c for c in colors],
        )
        ax.set_xticks(range(len(table)), table["label"], rotation=90, fontsize=7)
        table.to_csv(path.with_suffix(".csv"), index=False)
    elif kind == "line":
        rows = []
        for series in data["series"]:
            points = series["points"]
            color = series.get("color")
            xs, ys = [p["x"] for p in points], [p["y"] for p in points]
            if labels.get("step"):
                width = float(np.median(np.diff(np.sort(xs)))) if len(xs) > 1 else 1.0
                ax.bar(xs, ys, width=width, label=series.get("name", ""), color=color)
            else:
                ax.plot(xs, ys, label=series.get("name", ""), color=color)
            rows.extend({"series": series.get("name", ""), **p} for p in points)
        if len(data["series"]) > 1:
            ax.legend(fontsize=6)
        pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    elif kind == "scatter":
        table = pd.DataFrame(data["points"])
        categories = {c["name"]: c.get("color") for c in data.get("categories", [])}
        groups = (
            table.groupby("category", sort=False)
            if "category" in table
            else [("", table)]
        )
        for name, group in groups:
            ax.scatter(
                group["x"],
                group["y"],
                s=8,
                linewidths=0,
                label=name,
                color=categories.get(name),
                rasterized=True,
            )
        for point in data["points"]:
            if point.get("label"):
                ax.annotate(point["label"], (point["x"], point["y"]), fontsize=5)
        if categories:
            ax.legend(fontsize=6)
        for axis, enabled in data.get("log_scale", {}).items():
            if enabled:
                getattr(ax, f"set_{axis}scale")("log")
        table.to_csv(path.with_suffix(".csv"), index=False)
    elif kind in ("kde", "histogram", "boxplot"):
        groups = data["groups"]
        rows = []
        for group in groups:
            values = np.asarray(group["values"], dtype=float)
            values = values[np.isfinite(values)]
            label = group.get("label", "")
            rows.extend({"group": label, "value": float(v)} for v in values)
            if kind == "kde":
                grid = np.linspace(values.min(), values.max(), 200)
                ax.plot(grid, gaussian_kde(values)(grid), label=label)
            elif kind == "histogram":
                ax.hist(values, bins=25, alpha=0.5, label=label)
        if kind == "boxplot":
            ax.boxplot(
                [g["values"] for g in groups],
                tick_labels=[g.get("label", "") for g in groups],
                showfliers=False,
            )
        elif len(groups) > 1:
            ax.legend(fontsize=6)
        pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    elif kind == "quiver":
        ax.quiver(
            data["x"],
            data["y"],
            data["u"],
            data["v"],
            data["magnitude"],
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        pd.DataFrame({k: data[k] for k in ("x", "y", "u", "v", "magnitude")}).to_csv(
            path.with_suffix(".csv"), index=False
        )
    ax.set_title(title, fontsize=10)
    for axis in ("x", "y"):
        getattr(ax, f"set_{axis}label")(labels.get(f"{axis}label", ""))
        lower, upper = labels.get(f"{axis}min"), labels.get(f"{axis}max")
        if lower is not None or upper is not None:
            getattr(ax, f"set_{axis}lim")(lower, upper)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
