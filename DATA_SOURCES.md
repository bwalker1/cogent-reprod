# Dataset download links

Download public expression data from the original repositories below. Paths are
relative to `--data-dir` (usually `data/`).

## MOSTA: mouse organogenesis

Source: [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/).

Use per-section `.MOSTA.h5ad` files under **Embryo data**. Training selects
E1S1–E1S4 at E12.5–E16.5; figure analyses use E1S1 at each stage. Comparison additionally uses E14.5_E2S1.

| Save as | Download |
| --- | --- |
| `mosta/raw/E12.5_E1S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S1.MOSTA.h5ad) |
| `mosta/raw/E12.5_E1S2.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S2.MOSTA.h5ad) |
| `mosta/raw/E12.5_E1S3.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S3.MOSTA.h5ad) |
| `mosta/raw/E12.5_E1S4.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E12.5_E1S4.MOSTA.h5ad) |
| `mosta/raw/E13.5_E1S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S1.MOSTA.h5ad) |
| `mosta/raw/E13.5_E1S2.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S2.MOSTA.h5ad) |
| `mosta/raw/E13.5_E1S3.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S3.MOSTA.h5ad) |
| `mosta/raw/E13.5_E1S4.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E13.5_E1S4.MOSTA.h5ad) |
| `mosta/raw/E14.5_E1S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S1.MOSTA.h5ad) |
| `mosta/raw/E14.5_E1S2.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S2.MOSTA.h5ad) |
| `mosta/raw/E14.5_E1S3.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S3.MOSTA.h5ad) |
| `mosta/raw/E14.5_E1S4.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E1S4.MOSTA.h5ad) |
| `mosta/raw/E15.5_E1S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S1.MOSTA.h5ad) |
| `mosta/raw/E15.5_E1S2.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S2.MOSTA.h5ad) |
| `mosta/raw/E15.5_E1S3.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S3.MOSTA.h5ad) |
| `mosta/raw/E15.5_E1S4.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E15.5_E1S4.MOSTA.h5ad) |
| `mosta/raw/E16.5_E1S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S1.MOSTA.h5ad) |
| `mosta/raw/E16.5_E1S2.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S2.MOSTA.h5ad) |
| `mosta/raw/E16.5_E1S3.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S3.MOSTA.h5ad) |
| `mosta/raw/E16.5_E1S4.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S4.MOSTA.h5ad) |
| `mosta/raw/E14.5_E2S1.MOSTA.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E14.5_E2S1.MOSTA.h5ad) |

## Kidney carcinoma

Source: [CZ CELLxGENE collection](https://cellxgene.cziscience.com/collections/f7cecffa-00b4-4560-a29a-8ad626b8ee08).

These are the dataset versions recorded in the manuscript input files, associated
with [Li et al., Cancer Cell](https://doi.org/10.1016/j.ccell.2022.11.001).
Rename each download as shown.

| Save as | Download |
| --- | --- |
| `kidney-carcinoma/raw/visium_core.h5ad` | [H5AD](https://datasets.cellxgene.cziscience.com/99d31e0c-4ccf-42c4-8246-c9636d422c0f.h5ad) |
| `kidney-carcinoma/raw/visium_interface.h5ad` | [H5AD](https://datasets.cellxgene.cziscience.com/ec370e4c-4ce8-4354-afe4-9b4df651ca3d.h5ad) |
| `kidney-carcinoma/raw/singlecell.h5ad` | [H5AD](https://datasets.cellxgene.cziscience.com/4fd0f389-89ac-4a6a-a2ed-309d45ce6514.h5ad) |

## ARTISTA: axolotl telencephalon regeneration

Source: [ARTISTA downloads](https://db.cngb.org/stomics/artista/download/).

Use individual section H5ADs for 2, 5, 10, 15, and 20 DPI. After downloading, run
`python -m cogent.prepare_artista --data-dir data` to create injury-region labels.

| Save as | Download |
| --- | --- |
| `artista/raw/2DPI_1.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/2DPI_1.h5ad) |
| `artista/raw/2DPI_2.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/2DPI_2.h5ad) |
| `artista/raw/2DPI_3.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/2DPI_3.h5ad) |
| `artista/raw/5DPI_1.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/5DPI_1.h5ad) |
| `artista/raw/5DPI_2.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/5DPI_2.h5ad) |
| `artista/raw/5DPI_3.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/5DPI_3.h5ad) |
| `artista/raw/10DPI_1.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/10DPI_1.h5ad) |
| `artista/raw/10DPI_2.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/10DPI_2.h5ad) |
| `artista/raw/10DPI_3.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/10DPI_3.h5ad) |
| `artista/raw/15DPI_1.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/15DPI_1.h5ad) |
| `artista/raw/15DPI_2.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/15DPI_2.h5ad) |
| `artista/raw/15DPI_3.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/15DPI_3.h5ad) |
| `artista/raw/15DPI_4.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/15DPI_4.h5ad) |
| `artista/raw/20DPI_1.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/20DPI_1.h5ad) |
| `artista/raw/20DPI_2.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/20DPI_2.h5ad) |
| `artista/raw/20DPI_3.h5ad` | [H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/20DPI_3.h5ad) |

## Human CRC: Visium HD, 8-micron bins

Sources: [GEO GSE280315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315)
and [10x Genomics CRC dataset](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc).
The direct links provide the **processed 8-micron filtered count matrices and
spatial coordinates** for all five samples.
| Sample | Count matrix | Spatial coordinates |
| --- | --- | --- |
| P1 Cancer | [H5](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594567/suppl/GSM8594567_P1CRC_filtered_feature_bc_matrix.h5) | [Parquet, gzip](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594567/suppl/GSM8594567_P1CRC_tissue_positions.parquet.gz) |
| P2 Cancer | [H5](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594568/suppl/GSM8594568_P2CRC_filtered_feature_bc_matrix.h5) | [Parquet, gzip](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594568/suppl/GSM8594568_P2CRC_tissue_positions.parquet.gz) |
| P5 Cancer | [H5](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594569/suppl/GSM8594569_P5CRC_filtered_feature_bc_matrix.h5) | [Parquet, gzip](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594569/suppl/GSM8594569_P5CRC_tissue_positions.parquet.gz) |
| P3 Normal | [H5](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594570/suppl/GSM8594570_P3NAT_filtered_feature_bc_matrix.h5) | [Parquet, gzip](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594570/suppl/GSM8594570_P3NAT_tissue_positions.parquet.gz) |
| P5 Normal | [H5](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594571/suppl/GSM8594571_P5NAT_filtered_feature_bc_matrix.h5) | [Parquet, gzip](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594571/suppl/GSM8594571_P5NAT_tissue_positions.parquet.gz) |

These processed outputs need a local conversion to the H5AD format used here.
For the figure scripts, combine each count matrix with its barcode-matched
positions, retain `in_tissue`, and store full-resolution coordinates in
`obsm['spatial']` in `(pxl_col_in_fullres, pxl_row_in_fullres)` order. Keep gene
symbols as `var_names` and leave expression as counts. Save the resulting files as:

```text
visium_crc/raw/Visium_HD_Human_Colon_Cancer_P1_square_008um.h5ad
visium_crc/raw/Visium_HD_Human_Colon_Cancer_P2_square_008um.h5ad
visium_crc/raw/Visium_HD_Human_Colon_Cancer_P5_square_008um.h5ad
visium_crc/raw/Visium_HD_Human_Colon_Normal_P3_square_008um.h5ad
visium_crc/raw/Visium_HD_Human_Colon_Normal_P5_square_008um.h5ad
```
