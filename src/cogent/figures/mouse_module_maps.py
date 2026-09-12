"""The shared stage/module grid for the two mouse map supplements."""

import pickle
from pathlib import Path

import anndata as ad
import numpy as np

from cogent import Cogent
from cogent.utils import cluster_means_all
from cogent.figures.plotting import save_panel, spatial_panel


def calculate(data_dir, resolution=0.5, dataset="E12.5_E1S1", model=None):
    data_dir = Path(data_dir)
    if model is None:
        model = Cogent.load(data_dir / "mosta/models/cogent", device="cpu")
    adata = ad.read_h5ad(data_dir / f"mosta/raw/{dataset}.MOSTA.h5ad")
    means = cluster_means_all(model, adata, view="average", resolution=resolution)
    return {
        "spatial_xy": np.asarray(adata.obsm["spatial"], dtype=np.float32),
        "cluster_means": {
            int(k): np.asarray(v, dtype=np.float32) for k, v in means.items()
        },
    }


def run_maps(data_dir, output_dir, clusters):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Cogent.load(data_dir / "mosta/models/cogent", device="cpu")
    for index, stage in enumerate(
        ("E12.5", "E13.5", "E14.5", "E15.5", "E16.5"), start=1
    ):
        results = calculate(data_dir, dataset=f"{stage}_E1S1", model=model)
        with (output_dir / f"analysis_{index}.pkl").open("wb") as handle:
            pickle.dump(results, handle, protocol=5)
        suffix = stage.lower().replace(".", "")
        for cluster in clusters:
            save_panel(
                spatial_panel(results["spatial_xy"], results["cluster_means"][cluster]),
                "spatial",
                output_dir / f"s2-c{cluster}-{suffix}",
                stage,
                {},
            )
        del results
