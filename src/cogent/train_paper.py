"""Train the models for mouse development, kidney cancer, CRC, and regeneration."""

import argparse
from pathlib import Path
import scanpy as sc
from cogent.training import train


PAPER_SETTINGS = {
    "mosta": dict(
        spatial_radii=[1.5, 3.0],
        rand_embeddings=False,
        group_order=["12", "13", "14", "15", "16"],
    ),
    "kidney-carcinoma": dict(
        spatial_radii=[1.5, 2.5, 3.0],
        rand_embeddings=True,
        group_order=["cancer", "healthy"],
    ),
    "kidney-sc": dict(
        spatial_radii=[1.5, 2.5, 3.0],
        rand_embeddings=True,
        group_order=["cancer", "healthy"],
    ),
    "visium_crc": dict(
        spatial_radii=[44.0, 73.0, 88.0],
        rand_embeddings=True,
        group_order=["CRC", "NAT"],
    ),
    "artista": dict(
        spatial_radii=[42.0, 70.0, 84.0],
        rand_embeddings=True,
        group_order=["10DPI", "15DPI", "20DPI", "2DPI", "5DPI"],
    ),
}


def load_samples(data_dir, dataset):
    folder = Path(data_dir) / (
        "kidney-carcinoma" if dataset == "kidney-sc" else dataset
    )
    genes = (folder / "genes.txt").read_text().splitlines()
    if dataset == "mosta":
        files = [
            (p.name, str(stage))
            for stage in [12, 13, 14, 15, 16]
            for section in range(1, 5)
            for p in [folder / "raw" / f"E{stage}.5_E1S{section}.MOSTA.h5ad"]
            if p.exists()
        ]
    elif dataset == "artista":
        files = [
            (p.name, p.stem.split("_")[0])
            for p in sorted((folder / "raw").glob("*DPI_*.h5ad"))
            if int(p.name.split("DPI")[0]) in [2, 5, 10, 15, 20]
        ]
    elif dataset == "visium_crc":
        files = [
            (f"Visium_HD_Human_Colon_{tissue}_{patient}_square_008um.h5ad", group)
            for tissue, patient, group in [
                ("Cancer", "P1", "CRC"),
                ("Cancer", "P2", "CRC"),
                ("Cancer", "P5", "CRC"),
                ("Normal", "P3", "NAT"),
                ("Normal", "P5", "NAT"),
            ]
        ]
    else:
        files = [
            (name, None)
            for name in (
                ["singlecell.h5ad"]
                if dataset == "kidney-sc"
                else ["visium_core.h5ad", "visium_interface.h5ad", "singlecell.h5ad"]
            )
        ]
    if not files:
        raise FileNotFoundError(f"No samples found for {dataset} in {folder / 'raw'}")
    adatas = []
    for filename, group in files:
        a = sc.read_h5ad(folder / "raw" / filename)
        if dataset in ("kidney-carcinoma", "kidney-sc"):
            condition = (
                a.obs["disease"]
                .astype(str)
                .map(
                    {"normal": "healthy", "nonpapillary renal cell carcinoma": "cancer"}
                )
            )
            a = a[condition.notna()].copy()
            a.obs["group"] = condition.dropna().astype(str)
            if filename == "singlecell.h5ad":
                a.obs["sample"] = a.obs["orig.ident"].astype(str)
                a.obsm.pop("spatial", None)
                a.obsm.pop("X_spatial", None)
        else:
            if dataset == "visium_crc":
                a = a[a.obs["in_tissue"] == 1].copy()
            a.obs["group"] = group
            a.obs["sample"] = Path(filename).stem
        adatas.append(a)
    common = set.intersection(*(set(a.var_names) for a in adatas))
    return adatas, [g for g in genes if g in common]


def main():
    parser = argparse.ArgumentParser(description="Train a spatial gene embedding model")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mosta", "kidney-carcinoma", "kidney-sc", "visium_crc", "artista"],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batches-per-epoch", default=1000, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()
    adatas, genes = load_samples(args.data_dir, args.dataset)
    train(
        adatas,
        genes,
        args.output_dir,
        **PAPER_SETTINGS[args.dataset],
        beta=1.0,
        normalized=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        restart_period=10,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
