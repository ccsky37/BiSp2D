# BiSp2D

BiSp2D constructs the exact co-occurrence matrix `V^T V` for a binary incidence matrix on NVIDIA GPUs. The input is compressed into a 64-bit BaSC representation and processed by a symmetric BaSC-GEMM kernel.

This repository contains only the BiSp2D implementation and its random-input benchmark. External baselines and paper-specific experiment scripts are not included.

## Build

Requirements:

- NVIDIA GPU
- CUDA 12 or later
- C++17

```bash
make ARCH=80
```

`ARCH=80` targets A100. Set `ARCH` to the compute capability of another GPU when needed.

## Run

```bash
./run.sh [rows] [cols] [sparsity_percent] [g] [warmups] [runs]
```

The default command is:

```bash
./run.sh 20000 4096 90 0 5 5
```

`g=0` selects the default warp-based BaSC builder. Values `1`, `2`, `4`, `8`, `16`, and `32` select the grouped builder with the corresponding sub-vector group size.

For sparsity greater than 99%, BaSC-GEMM uses zero skipping and the forced software single-bit counter. Other inputs use native `__popcll()`.

The reported GPU total covers H2D transfer, BaSC construction, and BaSC-GEMM. Random input generation and result validation are excluded from timing.

## A100 results

NVIDIA A100 PCIe 40 GB, CUDA 12.4, `g=0`, five warm-up runs, and five measured runs:

| Matrix | Sparsity | H2D (ms) | Build BaSC (ms) | BaSC-GEMM (ms) | GPU Total (ms) |
|---|---:|---:|---:|---:|---:|
| 10K x 4096 | 90.0% | 3.941 | 0.049 | 1.204 | 5.180 |
| 20K x 4096 | 90.0% | 7.717 | 0.072 | 2.357 | 10.132 |
| 40K x 4096 | 90.0% | 15.359 | 0.108 | 4.694 | 20.152 |
| 80K x 4096 | 90.0% | 29.463 | 0.191 | 9.354 | 38.996 |
| 10K x 4096 | 99.5% | 3.811 | 0.050 | 1.059 | 4.906 |
| 20K x 4096 | 99.5% | 7.803 | 0.073 | 2.052 | 9.914 |
| 40K x 4096 | 99.5% | 15.548 | 0.112 | 4.038 | 19.685 |
| 80K x 4096 | 99.5% | 29.453 | 0.191 | 8.018 | 37.650 |

H2D and Build BaSC overlap across four CUDA streams. GPU Total is measured on the critical path and is not the arithmetic sum of the separately reported phase values.
