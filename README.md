# CoGent

CoGent learns condition-dependent gene embeddings from spatial gene expression.
This repository contains model training and the analyses for Figures 2–6 and the
supplementary figures.

## Installation

Use Python 3.12 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.txt .
```

`requirements.txt` pins the direct dependencies used for reproduction. Figure
analyses run on CPU; training supports CPU and NVIDIA GPUs.

## Data

[DATA_SOURCES.md](DATA_SOURCES.md) lists downloads for MOSTA mouse organogenesis,
kidney carcinoma, human CRC, and ARTISTA axolotl regeneration, including sample
selections and format-conversion instructions.

The figure analyses also need the manuscript checkpoints, gene lists, module
annotation CSVs, and comparator embeddings. These study-specific files are
separate from the public expression datasets and are not included in this
repository. Arrange the inputs as follows:

```text
data/
  mosta/
    raw/*.MOSTA.h5ad
    genes.txt
    musegnn/*_rna_expression.h5ad
    musegnn/*_pvalue.h5
    models/cogent/{model.pt,cluster_labels_res0.5.csv}
    models/comparison/{musegnn,scetm}.zarr
  kidney-carcinoma/
    raw/{visium_core,visium_interface,singlecell}.h5ad
    genes.txt
    models/cogent/{model.pt,cluster_labels_res0.5.csv}
    models/cogent_sc/model.pt
  visium_crc/
    raw/Visium_HD_Human_Colon_*_square_008um.h5ad
    genes.txt
    models/cogent/{model.pt,cluster_labels_res0.5.csv}
  artista/
    raw/*DPI_*.h5ad
    preprocessed/*DPI_*.h5ad
    genes.txt
    models/cogent/{model.pt,cluster_labels_res1.0.csv}
```

MOSTA, ARTISTA, and kidney single-cell inputs contain log-expression in `X`;
CRC and kidney spatial inputs contain counts. scETM uses MOSTA's
`layers['count']`.

For ARTISTA, create the injury-region labels and `preprocessed/` files first:

```bash
python -m cogent.prepare_artista --data-dir data
```

## Reproduce the figures

Run all main and supplementary analyses:

```bash
python -m cogent.figures --data-dir data --output-dir results
```

To run one figure, use its module name:

```bash
python -m cogent.figures.fig2 --data-dir data --output-dir results
python -m cogent.figures.si_s04_mouse_developmental_shifts --data-dir data --output-dir results
```

The main modules are `fig2` through `fig6`. Supplementary modules are named
`si_s01_*` through `si_s11_*` in [src/cogent/figures](src/cogent/figures).

Each figure produces PDF panels, CSV tables, and pickle files containing the
analysis results and panel values. Rerunning replaces files in the chosen output
directory. Panels are exported individually; their styling may differ from the
manuscript. Pickle files retain the unscaled numerical values.

## Train the paper models

The [paper training script](src/cogent/train_paper.py) sets the sample selections,
condition labels, spatial radii, and model settings for each dataset:

```bash
python -m cogent.train_paper --data-dir data --dataset mosta \
  --output-dir refits/mosta --device cuda
```

Dataset choices are `mosta`, `kidney-carcinoma`, `kidney-sc`, `visium_crc`, and
`artista`. Use `--device cpu --num-workers 0` for CPU training. Training length,
batch size, and random seed can be set through the command-line options; see
`python -m cogent.train_paper --help`.

New fits can produce different embeddings and module assignments. The figure
scripts use modules from the manuscript checkpoints, so their annotations and
module selections need to be reassigned before using a new fit.

## Comparison methods

Install the extra dependencies and train MuSe-GNN and scETM for Figure 3:

```bash
python -m pip install -c requirements.txt '.[benchmarks]'
python -m cogent.benchmarks.train_musegnn --device cuda
python -m cogent.benchmarks.train_scetm --device cuda
```

MuSe-GNN reads the per-stage expression and coexpression-graph inputs; scETM reads
MOSTA AnnData files and fits each stage separately. The commands write
`musegnn.zarr` and `scetm.zarr` to `data/mosta/models/comparison/`. Use a different
`--output-dir` if those files already exist. Figure 3 reads these two stores and
computes PCA directly from the expression data.

## Use CoGent with your data

Each AnnData file needs raw counts in `X`, condition labels in `obs['group']`,
and tissue-section identifiers in `obs['sample']`. Put spatial coordinates in
`obsm['spatial']`; samples without coordinates use expression alone. List the
shared genes, one per line, in `genes.txt`.

```bash
cogent-train sample1.h5ad sample2.h5ad --genes genes.txt \
  --radii 1.5 2.5 3.0 --output-dir trained_model --device cuda
```

Radii use the units of your coordinates. Use `--normalized` for expression that
has already been normalized and log-transformed, or `--layer counts` to read a
count layer. Gene-wise scaling is applied in either case. Training saves
`model.pt`, module annotation CSVs, `training_loss.csv`, and
`training_parameters.json`.

Load a trained model and get its gene embeddings:

```python
from cogent import Cogent

model = Cogent.load("trained_model", device="cpu")
embeddings = model.get_embeddings()[..., :model.model.embedding_dim]
```
