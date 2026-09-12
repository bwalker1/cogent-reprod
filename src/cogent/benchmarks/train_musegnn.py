"""Train MuSe-GNN on MOSTA data for comparison with Cogent.

Reproduces the MuSe-GNN training pipeline from:
  "MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data"
  (HelloWorldLTY/MuSe-GNN)

Loads MOSTA expression and coexpression graphs and saves gene embeddings.
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch_geometric.data as pyg_data
import torch_geometric.nn
import zarr
from cogent.benchmarks.outputs import output_paths
from pytorch_metric_learning import losses
from torch_geometric.nn import TransformerConv
from cogent.benchmarks import gene_pairs as process_pair

STAGES = [
    ("E12_5_E1S1", "E12.5"),
    ("E13_5_E1S1", "E13.5"),
    ("E14_5_E2S1", "E14.5"),
    ("E15_5_E1S1", "E15.5"),
    ("E16_5_E1S1", "E16.5"),
]
DEFAULT_DATA_DIR = Path("data/mosta/musegnn")
DEFAULT_GENES_PATH = Path("data/mosta/genes.txt")
DEFAULT_OUTPUT_DIR = Path("data/mosta/models/comparison")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GCNEncoder_Multiinput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Multiinput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        conv_dict = {}
        conv_dict1 = {}
        for i in graph_list:
            conv_dict[i.show_index] = torch_geometric.nn.Sequential(
                "x, edge_index",
                [
                    (
                        TransformerConv(i.x.shape[1], out_channels, heads=4),
                        "x, edge_index-> x",
                    ),
                    (torch_geometric.nn.GraphNorm(out_channels * 4), "x -> x"),
                ],
            )
        self.convl1 = nn.ModuleDict(conv_dict)
        self.convl2 = nn.ModuleDict(conv_dict1)

    def forward(self, x, edge_index, show_index):
        x = self.convl1[show_index](x, edge_index)
        x = self.activ(x)
        return x


class GCNEncoder_Commoninput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Commoninput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        conv_dict_l2 = {}
        conv_dict_l3 = {}
        conv_dict_l4 = {}
        tissue_specific_list = list(set(label_list))
        for i in tissue_specific_list:
            conv_dict_l2[i] = torch_geometric.nn.Sequential(
                "x, edge_index",
                [
                    (
                        TransformerConv(out_channels * 4, out_channels, heads=2),
                        "x, edge_index -> x",
                    ),
                    (torch_geometric.nn.GraphNorm(out_channels * 2), "x -> x"),
                ],
            )
            conv_dict_l3[i] = TransformerConv(out_channels * 2, out_channels)
            conv_dict_l4[i] = TransformerConv(out_channels * 4, out_channels)
        self.convl2 = nn.ModuleDict(conv_dict_l2)
        self.convl3 = nn.ModuleDict(conv_dict_l3)
        self.convl4 = nn.ModuleDict(conv_dict_l4)

    def forward(self, x, edge_index, show_index):
        x_inp = x
        x = self.convl2[show_index.split("__")[0]](x, edge_index)
        x = self.activ(x)
        x = self.convl3[show_index.split("__")[0]](x, edge_index)
        return x + self.convl4[show_index.split("__")[0]](x_inp, edge_index)


class MLP_edge_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, graph_list):
        super(MLP_edge_Decoder, self).__init__()
        dec_dict = {}
        self.activ = nn.Mish(inplace=True)
        for i in graph_list:
            dec_dict[i.show_index] = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels),
                self.activ,
                nn.Linear(in_channels, out_channels),
                self.activ,
                nn.Linear(in_channels, out_channels),
            )
        self.MLP = nn.ModuleDict(dec_dict)

    def forward(self, x, show_index):
        x = self.MLP[show_index](x)
        return x


def load_mosta_graphs(data_dir, genes_path):
    """Load preprocessed MOSTA data into MuSe-GNN graph format.

    Returns: graph_list, cor_list, label_list, graph_networkx_list, gene_names
    """
    with open(genes_path) as f:
        gene_names = [line.strip() for line in f.readlines()]
    gene_index = pd.Index(gene_names)
    graph_list = []
    cor_list = []
    label_list = []
    graph_networkx_list = []
    for stage_file, stage_label in STAGES:
        expr_path = data_dir / f"mosta_{stage_file}_rna_expression.h5ad"
        pval_path = data_dir / f"mosta_{stage_file}_pvalue.h5"
        adata = sc.read_h5ad(expr_path)
        expression = adata.X.copy()
        if hasattr(expression, "toarray"):
            expression = expression.toarray()
        with h5py.File(pval_path, "r") as f:
            pvalue_matrix = f["df/block0_values"][:]
        correlation = pd.DataFrame(pvalue_matrix, index=gene_index, columns=gene_index)
        cor_list.append(correlation)
        edges_new = np.array(
            [np.nonzero(correlation.values)[0], np.nonzero(correlation.values)[1]]
        )
        graph = pyg_data.Data(
            x=torch.FloatTensor(expression),
            edge_index=torch.FloatTensor(edges_new).long(),
        )
        vis = nx.from_pandas_adjacency(correlation)
        graph_networkx_list.append(vis)
        graph.gene_list = gene_index
        graph.show_index = f"spatial_mosta__{stage_file}"
        graph_list.append(graph)
        label_list.append("spatial_mosta")
        print(
            f"Loaded {stage_label}: {expression.shape[0]} genes, {expression.shape[1]} cells, {edges_new.shape[1]} edges"
        )
    return (graph_list, cor_list, label_list, graph_networkx_list, gene_names)


def compute_gene_sets(graph_list, graph_networkx_list, num_threads=cpu_count()):
    common_gene_set = {}
    common_gene_overlap = {}
    diff_gene_set = {}
    diff_gene_neighbor_set = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_pair.process_pair, i, j, graph_list, graph_networkx_list
            )
            for i in range(len(graph_list))
            for j in range(len(graph_list))
            if i != j
        ]
        for future in as_completed(futures):
            (i, j, common_set, common_overlap, diff_set, diff_neighbor_set) = (
                future.result()
            )
            key = graph_list[i].show_index + graph_list[j].show_index
            common_gene_set[key] = common_set
            common_gene_overlap[key] = common_overlap
            diff_gene_set[key] = diff_set
            diff_gene_neighbor_set[key] = diff_neighbor_set
    return (common_gene_set, common_gene_overlap, diff_gene_set, diff_gene_neighbor_set)


def train_musegnn(
    graph_list,
    cor_list,
    label_list,
    common_gene_set,
    common_gene_overlap,
    diff_gene_set,
    diff_gene_neighbor_set,
    seed=0,
    epochs=2000,
    dim=32,
    lr_enc=0.0001,
    lr_dec=0.001,
    lambda_c=0.01,
    device_str="cuda",
):
    """Run MuSe-GNN training for a single seed. Returns embeddings (n_groups, n_genes, dim)."""
    device = torch.device(device_str)
    n_genes = graph_list[0].x.shape[0]
    set_seed(seed)
    gene_encoder_is = GCNEncoder_Multiinput(dim, graph_list, label_list).to(device)
    gene_encoder_com = GCNEncoder_Commoninput(dim, graph_list, label_list).to(device)
    gene_decoder = MLP_edge_Decoder(n_genes, n_genes, graph_list).to(device)
    optimizer_enc_is = torch.optim.Adam(gene_encoder_is.parameters(), lr=lr_enc)
    optimizer_enc_com = torch.optim.Adam(gene_encoder_com.parameters(), lr=lr_enc)
    optimizer_dec2 = torch.optim.Adam(gene_decoder.parameters(), lr=lr_dec)
    loss_f = nn.BCEWithLogitsLoss()
    loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss())
    lambda_infonce = lambda_c
    graph_index_list = list(range(len(graph_list)))
    edge_adj_list = [
        torch.FloatTensor(cor_list[i].values).to(device) for i in graph_index_list
    ]

    def penalize_data(z, graph, graph_list, j):
        loss = torch.tensor(0.0).to(device)
        graph_new = graph_list[j]
        x = graph_new.x.to(device)
        train_pos_edge_index = graph_new.edge_index.to(device)
        x = gene_encoder_is(x, train_pos_edge_index, graph_new.show_index)
        z_new = gene_encoder_com(x, train_pos_edge_index, graph_new.show_index)
        [index_i, index_j] = common_gene_set[graph.show_index + graph_new.show_index]
        if len(index_i) == 0 or len(index_j) == 0:
            return loss
        z_cor = z[index_i]
        z_new_cor = z_new[index_j]
        weight = torch.FloatTensor(
            common_gene_overlap[graph.show_index + graph_new.show_index]
        ).to(device)
        cos_sim = torch.cosine_similarity(z_cor, z_new_cor, axis=1) * weight
        loss += -cos_sim.mean()
        [index_i, index_j] = diff_gene_set[graph.show_index + graph_new.show_index]
        if len(index_i) == 0 or len(index_j) == 0:
            return loss
        opt_index = np.random.choice(
            [i for i in range(len(index_i))], min(100, len(index_i))
        )
        z_diff = z[index_i[opt_index]]
        z_new_diff = z_new[index_j[opt_index]]
        [index_i, index_j] = diff_gene_neighbor_set[
            graph.show_index + graph_new.show_index
        ]
        z_diff_true = z[index_i[opt_index]]
        z_new_diff_true = z_new[index_j[opt_index]]
        loss += lambda_infonce * loss_func(
            torch.cat((z_diff, z_new_diff)), torch.cat((z_diff_true, z_new_diff_true))
        )
        return loss

    import time

    gene_encoder_is.train()
    gene_encoder_com.train()
    t_train_start = time.time()
    for epoch in range(epochs):
        for i in range(len(graph_index_list)):
            optimizer_enc_is.zero_grad(set_to_none=True)
            optimizer_enc_com.zero_grad(set_to_none=True)
            optimizer_dec2.zero_grad(set_to_none=True)
            graph = graph_list[i]
            x = graph.x.to(device)
            train_pos_edge_index = graph.edge_index.to(device)
            edge_adj = edge_adj_list[i]
            x = gene_encoder_is(x, train_pos_edge_index, graph.show_index)
            z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)
            adj = torch.matmul(z, z.t())
            edge_reconstruct = gene_decoder(adj, graph.show_index)
            loss = loss_f(edge_reconstruct.flatten(), edge_adj.flatten())
            if epoch % 200 == 0:
                print(
                    f"  seed={seed} epoch={epoch} graph={i} recon_loss={loss.item():.4f}"
                )
            graph_index_list_copy = graph_index_list.copy()
            graph_index_list_copy.remove(i)
            j = random.sample(graph_index_list_copy, 1)
            loss += penalize_data(z, graph, graph_list, j[0])
            del graph
            loss.backward()
            del loss
            optimizer_enc_is.step()
            optimizer_enc_com.step()
            optimizer_dec2.step()
        if epoch % 200 == 0:
            elapsed = time.time() - t_train_start
            per_epoch = elapsed / (epoch + 1)
            print(
                f"  epoch {epoch} finish ({elapsed:.1f}s elapsed, {per_epoch:.2f}s/epoch)"
            )
    torch.cuda.empty_cache()
    emb_list = []
    gene_encoder_is.eval()
    gene_encoder_com.eval()
    with torch.no_grad():
        for i in range(len(graph_list)):
            g = graph_list[i]
            x = g.x.to(device)
            train_pos_edge_index = g.edge_index.long().to(device)
            x = gene_encoder_is(x, train_pos_edge_index, g.show_index)
            z = gene_encoder_com(x, train_pos_edge_index, g.show_index)
            emb_list.append(z.cpu().numpy())
    embeddings = np.stack(emb_list)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Train MuSe-GNN on MOSTA data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lr-enc", type=float, default=0.0001)
    parser.add_argument("--lr-dec", type=float, default=0.001)
    parser.add_argument(
        "--lambda-c",
        type=float,
        default=0.01,
        help="Contrastive weight for genes unique to a graph; inactive for the shared-gene MOSTA inputs",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--genes-path", type=Path, default=DEFAULT_GENES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = output_paths(args.output_dir, "musegnn", args.seeds)
    import time

    t0 = time.time()
    print("Loading MOSTA data...")
    (graph_list, cor_list, label_list, graph_networkx_list, gene_names) = (
        load_mosta_graphs(args.data_dir, args.genes_path)
    )
    t_load = time.time()
    print(f"Data loading: {t_load - t0:.1f}s")
    print("Computing gene sets...")
    (common_gene_set, common_gene_overlap, diff_gene_set, diff_gene_neighbor_set) = (
        compute_gene_sets(graph_list, graph_networkx_list)
    )
    t_preprocess = time.time()
    print(f"Gene set computation: {t_preprocess - t_load:.1f}s")
    print(f"Total preprocessing: {t_preprocess - t0:.1f}s")
    group_ids = [label for (_, label) in STAGES]
    for seed, zarr_path in zip(args.seeds, paths):
        print(f"\n=== Training seed {seed} ===")
        embeddings = train_musegnn(
            graph_list,
            cor_list,
            label_list,
            common_gene_set,
            common_gene_overlap,
            diff_gene_set,
            diff_gene_neighbor_set,
            seed=seed,
            epochs=args.epochs,
            dim=args.dim,
            lr_enc=args.lr_enc,
            lr_dec=args.lr_dec,
            lambda_c=args.lambda_c,
            device_str=args.device,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        store = zarr.open_group(str(zarr_path), mode="w-")
        store.attrs.update(dict(seed=seed, mode="joint"))
        store["embeddings"] = embeddings
        store["gene_names"] = np.array(gene_names)
        store["group_ids"] = np.array(group_ids)
        print(f"Saved embeddings to {zarr_path}: shape {embeddings.shape}")


if __name__ == "__main__":
    main()
