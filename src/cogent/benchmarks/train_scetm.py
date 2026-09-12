"""Train scETM on MOSTA data for comparison with Cogent.

Uses the scETM package (single-cell Embedded Topic Model) from:
  "Learning interpretable cellular and gene signature embeddings from
   single-cell transcriptomic data" (Nature Communications, 2021)
  https://github.com/hui2000ji/scETM

Trains independent scETM models for the five developmental stages, as in Figure 3.
"""

import argparse
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import scanpy as sc
import torch
import zarr
from cogent.benchmarks.outputs import output_paths

torch.distributions.Distribution.set_default_validate_args(False)
STAGES = [
    ("E12.5_E1S1", "E12.5"),
    ("E13.5_E1S1", "E13.5"),
    ("E14.5_E2S1", "E14.5"),
    ("E15.5_E1S1", "E15.5"),
    ("E16.5_E1S1", "E16.5"),
]
DEFAULT_DATA_DIR = Path("data/mosta/raw")
DEFAULT_GENES_PATH = Path("data/mosta/genes.txt")
DEFAULT_OUTPUT_DIR = Path("data/mosta/models/comparison")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mosta_stages(data_dir, genes_path):
    """Load MOSTA stages, subset to genes, use raw counts in X.

    Returns: list of (adata, stage_label) tuples, gene_names list
    """
    with open(genes_path) as f:
        gene_names = [line.strip() for line in f.readlines()]
    stages = []
    for stage_file, stage_label in STAGES:
        path = data_dir / f"{stage_file}.MOSTA.h5ad"
        adata = sc.read_h5ad(path)
        gene_mask = adata.var_names.isin(gene_names)
        adata = adata[:, gene_mask].copy()
        adata = adata[:, gene_names].copy()
        adata.X = adata.layers["count"].astype(np.float32).copy()
        lib_size = np.array(adata.X.sum(axis=1)).flatten()
        nonzero_mask = lib_size > 0
        n_removed = (~nonzero_mask).sum()
        if n_removed > 0:
            adata = adata[nonzero_mask].copy()
            print(f"  Removed {n_removed} cells with zero library size")
        adata.obs["batch_indices"] = 0
        adata.obs["cell_types"] = "unknown"
        print(f"Loaded {stage_label}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        stages.append((adata, stage_label))
    return (stages, gene_names)


def train_scetm_on_adata(
    adata,
    n_topics,
    emb_dim,
    epochs,
    device,
    seed,
    lr=0.005,
):
    """Train scETM on an AnnData and return the model."""
    from scETM import scETM as ScETMModel
    from scETM import UnsupervisedTrainer

    set_seed(seed)
    model = ScETMModel(
        n_trainable_genes=adata.shape[1],
        n_batches=1,
        n_topics=n_topics,
        trainable_gene_emb_dim=emb_dim,
        enable_batch_bias=False,
        input_batch_id=False,
        normed_loss=True,
        norm_cells=True,
        device=device,
    )
    with TemporaryDirectory(prefix="scetm-") as temporary:
        trainer = UnsupervisedTrainer(
            model=model,
            adata=adata,
            ckpt_dir=temporary,
            test_ratio=0.1,
            seed=seed,
            init_lr=lr,
        )
        trainer.train(n_epochs=epochs, eval=False, save_model=False)
    return model


def train_per_stage(stages, n_topics, emb_dim, epochs, device, seed, lr=0.001):
    """Train independent scETM per stage. Returns (5, n_genes, emb_dim)."""
    import time

    rho_list = []
    for adata, stage_label in stages:
        print(f"\n  Training scETM for {stage_label}...")
        t_stage = time.time()
        model = train_scetm_on_adata(
            adata, n_topics, emb_dim, epochs, device, seed, lr=lr
        )
        elapsed = time.time() - t_stage
        per_epoch = elapsed / epochs
        rho = model.rho_trainable_emb.trainable.detach().cpu().numpy().T
        rho_list.append(rho)
        print(
            f"  {stage_label} rho shape: {rho.shape} ({elapsed:.1f}s, {per_epoch:.2f}s/epoch)"
        )
    embeddings = np.stack(rho_list)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Train scETM on MOSTA data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-topics", type=int, default=50)
    parser.add_argument("--emb-dim", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--genes-path", type=Path, default=DEFAULT_GENES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = output_paths(args.output_dir, "scetm", args.seeds, "per-stage")
    import time

    t0 = time.time()
    print("Loading MOSTA data...")
    (stages, gene_names) = load_mosta_stages(args.data_dir, args.genes_path)
    print(f"Data loading: {time.time() - t0:.1f}s")
    group_ids = [label for (_, label) in STAGES]
    for seed, zarr_path in zip(args.seeds, paths):
        print(f"\n=== Training seed {seed} ===")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        embeddings = train_per_stage(
            stages,
            args.n_topics,
            args.emb_dim,
            args.epochs,
            args.device,
            seed,
            lr=args.lr,
        )
        store = zarr.open_group(str(zarr_path), mode="w-")
        store.attrs.update(dict(seed=seed, mode="per-stage"))
        store["embeddings"] = embeddings
        store["gene_names"] = np.array(gene_names)
        store["group_ids"] = np.array(group_ids)
        print(f"Saved to {zarr_path}: embeddings shape {embeddings.shape}")


if __name__ == "__main__":
    main()
