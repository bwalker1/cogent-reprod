"""Train condition-dependent gene embeddings and save reproduction checkpoints."""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from cogent import Cogent
from cogent.dataset import CogentDataset, CogentIterableDataset
from cogent.model import Model


def train(
    adatas,
    genes,
    output_dir,
    *,
    condition_key="group",
    sample_key="sample",
    spatial_radii=(1.5, 2.5, 3.0),
    normalized=False,
    layer=None,
    embedding_dim=64,
    n_heads=1,
    decoder_depth=4,
    decoder_dim=64,
    cross_p=0.05,
    lambda_reg=0.001,
    beta=1.0,
    rand_embeddings=True,
    group_order=None,
    epochs=100,
    batches_per_epoch=1000,
    batch_size=512,
    gene_batch_size=512,
    learning_rate=0.01,
    restart_period=10,
    num_workers=0,
    device="cpu",
    seed=42,
):
    """Run spatial preprocessing, model fitting, UMAP, and Leiden clustering.

    Each AnnData needs condition labels; sample labels separate tissue sections.
    X is used directly when normalized=True. Otherwise the original per-cell
    normalization and log1p transform is applied before gene-wise scaling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for adata in adatas:
        if layer is not None:
            adata.X = adata.layers[layer].copy()
        if sample_key not in adata.obs:
            adata.obs[sample_key] = "sample"
        if "spatial" not in adata.obsm and "X_spatial" in adata.obsm:
            adata.obsm["spatial"] = adata.obsm["X_spatial"]
    dataset = CogentDataset(
        adatas=adatas,
        genes=list(genes),
        group_name=condition_key,
        sample_name=sample_key,
        cell_batch_size=batch_size,
        in_memory=False,
    )
    try:
        dataset.preprocess(spatial_conv=list(spatial_radii), normalize=not normalized)
        groups = dataset.group_ids
        if group_order is not None and groups != list(group_order):
            raise ValueError(f"Expected condition order {group_order}, got {groups}")
        parameters = dict(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            decoder_depth=decoder_depth,
            decoder_dim=decoder_dim,
            cross_p=cross_p,
            gene_batch_size=gene_batch_size,
            beta=beta,
            emb_dropout=0.0,
            rand_embeddings=rand_embeddings,
            lambda_reg=lambda_reg,
        )
        model = Model(
            n_genes=dataset.n_genes,
            n_groups=dataset.n_groups,
            n_channels=dataset.n_channels,
            **parameters,
        ).to(device)
        device_obj = torch.device(device)
        if device_obj.type == "cuda":
            torch.set_float32_matmul_precision("high")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, fused=device_obj.type == "cuda"
        )
        if restart_period is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=restart_period, T_mult=1
            )
        data_source = dataset
        if num_workers > 0:
            data_source = DataLoader(
                CogentIterableDataset(dataset._path),
                batch_size=None,
                num_workers=num_workers,
                pin_memory=device_obj.type == "cuda",
                prefetch_factor=10,
            )
        history = []
        schedule_counter = 0
        model.train()
        for epoch in range(epochs):
            total_loss, updates = 0.0, 0
            for batch_index, (group, expression, has_spatial) in enumerate(data_source):
                if batch_index >= batches_per_epoch:
                    break
                expression = torch.as_tensor(
                    expression, dtype=torch.float32, device=device
                )
                group = torch.as_tensor(group, dtype=torch.int64, device=device)
                for gene_start in range(0, dataset.n_genes, gene_batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(
                        expression=expression,
                        id=group,
                        gene_batch_idx=gene_start,
                        cross_genes=True,
                        has_spatial=bool(has_spatial),
                    )
                    loss.backward()
                    optimizer.step()
                    schedule_counter += 1
                    scheduler.step(epoch + schedule_counter / batches_per_epoch)
                    total_loss += float(loss.detach())
                    updates += 1
                with torch.no_grad():
                    interaction = model.embeddings[..., :embedding_dim]
                    renormed = torch.renorm(
                        interaction.reshape(-1, embedding_dim), 2, 0, 1 - model.eps
                    )
                    interaction.copy_(renormed.reshape_as(interaction))
            if not updates:
                raise ValueError(
                    "No training batches; reduce --batch-size for small inputs"
                )
            history.append(
                dict(
                    epoch=epoch + 1,
                    loss=total_loss / updates,
                    gene_batches=updates,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
            )
            print(f"Epoch {epoch + 1}: {history[-1]['loss']:.6f}", flush=True)
            # Preserve the training schedule's running update counter across epochs.
            schedule_counter *= 0.9
            scheduler.step(epoch + 1)
        model.eval()
    finally:
        dataset.close()
        if dataset._path is not None:
            shutil.rmtree(dataset._path)
    fitted = Cogent()
    fitted._model = model
    fitted._embeddings = model.embeddings.detach().cpu().numpy()
    fitted.compute_umap(random_state=seed)
    for resolution in (0.5, 1.0):
        labels = fitted.compute_leiden(resolution=resolution)
        ids, counts = np.unique(labels["average"], return_counts=True)
        pd.DataFrame(
            dict(cluster=ids, label=[f"Module {i}" for i in ids], n_genes=counts)
        ).to_csv(output_dir / f"cluster_labels_res{resolution}.csv", index=False)
    training_parameters = dict(
        seed=seed,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        restart_period=restart_period,
        num_workers=num_workers,
        spatial_radii=list(spatial_radii),
        normalized=normalized,
        layer=layer,
        condition_key=condition_key,
        sample_key=sample_key,
    )
    checkpoint = dict(
        model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        hyperparameters=parameters,
        training_parameters=training_parameters,
        metadata=dict(
            n_genes=dataset.n_genes,
            n_groups=dataset.n_groups,
            n_channels=dataset.n_channels,
            gene_names=dataset.gene_names,
            group_ids=groups,
        ),
        embeddings=fitted._embeddings,
        umap_coords=fitted._umap_coords,
        umap_intermediate=fitted._umap_intermediate,
        leiden_results=fitted._leiden_results,
        training_history=history,
    )
    (output_dir / "training_parameters.json").write_text(
        json.dumps({**training_parameters, **parameters}, indent=2) + "\n"
    )
    torch.save(checkpoint, output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_loss.csv", index=False)
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train condition-dependent gene embeddings"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--genes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-key", default="group")
    parser.add_argument("--sample-key", default="sample")
    parser.add_argument("--radii", nargs="*", type=float, default=[1.5, 2.5, 3.0])
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--layer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--rand-embeddings", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--restart-period", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(
        [sc.read_h5ad(p) for p in args.inputs],
        args.genes.read_text().splitlines(),
        args.output_dir,
        condition_key=args.condition_key,
        sample_key=args.sample_key,
        spatial_radii=args.radii,
        normalized=args.normalized,
        layer=args.layer,
        batch_size=args.batch_size,
        restart_period=args.restart_period,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        device=args.device,
        seed=args.seed,
        beta=args.beta,
        rand_embeddings=args.rand_embeddings,
    )


if __name__ == "__main__":
    main()
