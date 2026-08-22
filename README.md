# BiSp2D

BiSp2D constructs the exact co-occurrence matrix `V^T V` for a binary incidence matrix on NVIDIA GPUs. The input is compressed into a 64-bit BaSC representation and processed by a symmetric BaSC-GEMM kernel.

This repository contains only the BiSp2D implementation and its random-input benchmark. External baselines and paper-specific experiment scripts are not included.

## Dataset configurations

[`datasets/paper_datasets.csv`](datasets/paper_datasets.csv) contains the names, dimensions, and sparsity levels of the nine datasets reported in the paper. The included benchmark does not redistribute or load the original datasets. It generates binary random input with a fixed seed and the corresponding dimensions and sparsity, so the dataset names identify paper-scale configurations rather than copies of the real data.

The original datasets can be obtained from the sources cited in the paper:

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
- [Criteo AI Lab datasets](https://ailab.criteo.com/ressources/)
- [Anonymous Microsoft Web Data](https://kdd.ics.uci.edu/databases/msweb/msweb.html)
- [MovieLens](https://grouplens.org/datasets/movielens/)

Applications using real data can preprocess it into a row-major `int8_t` binary matrix and call `runBiSp2D()` directly.

## Build

Requirements:

- NVIDIA GPU
- CUDA 12 or later
- C++17
- Python 3 for the paper-scale table script

```bash
make ARCH=80
```

`ARCH=80` targets A100. Set `ARCH` to the compute capability of another GPU when needed.

## Run

```bash
./run.sh [rows] [cols] [sparsity_percent] [g] [warmups] [runs] [counter]
```

The default command is:

```bash
./run.sh 20000 4096 90 0 5 5 native
```

`g=0` selects the default warp-based BaSC builder. Values `1`, `2`, `4`, `8`, `16`, and `32` select the grouped builder with the corresponding sub-vector group size.

The default `counter=native` path uses `__popcll()`. For sparsity greater than 99%, it also skips all-zero operand groups. `counter=software` retains the data-dependent software single-bit counter while using the same sparsity-controlled zero-skipping rule.

The reported GPU total covers H2D transfer, BaSC construction, and BaSC-GEMM. Random input generation and result validation are excluded from timing.

## Paper-scale random benchmark

Run every configuration from `datasets/paper_datasets.csv` and print one Markdown table:

```bash
ARCH=80 ./run_paper_table.sh 5 5
```

The two optional arguments are the warm-up and measured-run counts.

## A100 results

NVIDIA A100 PCIe 40 GB, CUDA 12.4, `g=0`, five warm-up runs, and five measured runs:

| Dataset | Instances | Features | Sparsity | H2D (ms) | Device-side Work (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Amazon Fashion | 2,035,520 | 512 | 99.76% | 89.537 | 8.235 | 97.762 |
| Beauty & Personal Care | 2,000,000 | 1,024 | 99.66% | 172.246 | 16.280 | 188.512 |
| Clothing, Shoes & Jewelry | 1,516,928 | 2,048 | 99.68% | 260.097 | 40.575 | 300.650 |
| Amazon Subscription Boxes | 15,264 | 128 | 99.18% | 0.212 | 0.089 | 0.295 |
| Appliances | 89,664 | 160 | 99.38% | 1.242 | 0.339 | 1.576 |
| Beauty & Personal Care (meta) | 926,048 | 704 | 99.86% | 53.892 | 3.797 | 57.679 |
| Criteo | 200,000 | 224 | 88.39% | 3.753 | 0.576 | 4.323 |
| MSWeb | 37,728 | 288 | 98.95% | 0.952 | 0.132 | 1.077 |
| MovieLens | 50,000 | 2,000 | 99.71% | 8.309 | 1.279 | 9.576 |

H2D and Build BaSC overlap across four CUDA streams. GPU Total is measured on the critical path and is not the arithmetic sum of the separately reported phase values.

`Device-side Work` is Build BaSC plus BaSC-GEMM. These measurements use generated random inputs and therefore are paper-scale implementation checks, not replacements for results measured with the original datasets.
