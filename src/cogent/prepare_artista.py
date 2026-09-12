"""Prepare injury-region labels for the regeneration analyses."""

from pathlib import Path
import argparse
import anndata as ad
import pandas as pd


def prepare(data_dir):
    data_dir = Path(data_dir)
    output_dir = data_dir / "artista/preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((data_dir / "artista/raw").glob("*DPI_*.h5ad")):
        a = ad.read_h5ad(path)
        zone = pd.Series("uninj", index=a.obs_names)
        if "inj_M_L" in a.obs:
            zone[a.obs["inj_uninj"] == "inj"] = "inj_nonwound"
            zone[a.obs["inj_M_L"].isin(["inj_D_LP", "inj_D_MP"])] = "wound"
        else:
            zone[a.obs["inj_uninj"] == "inj"] = "wound"
        a.obs["inj_zone"] = pd.Categorical(
            zone, categories=["wound", "inj_nonwound", "uninj"]
        )
        a.write_h5ad(output_dir / path.name)
        print(path.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare regeneration tissue labels")
    parser.add_argument("--data-dir", type=Path, required=True)
    prepare(parser.parse_args().data_dir)
