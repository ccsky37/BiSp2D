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

BaSC-GEMM uses native `__popcll()`. For sparsity greater than 99%, it also skips all-zero operand groups.

The reported GPU total covers H2D transfer, BaSC construction, and BaSC-GEMM. Random input generation and result validation are excluded from timing.

## A100 results

NVIDIA A100 PCIe 40 GB, CUDA 12.4, `g=0`, five warm-up runs, and five measured runs:

| Matrix | Sparsity | H2D (ms) | Build BaSC (ms) | BaSC-GEMM (ms) | GPU Total (ms) |
|---|---:|---:|---:|---:|---:|
| 10K x 4096 | 90.0% | 3.799 | 0.059 | 1.706 | 5.549 |
| 20K x 4096 | 90.0% | 6.814 | 0.065 | 2.359 | 9.230 |
| 40K x 4096 | 90.0% | 15.066 | 0.111 | 4.694 | 19.860 |
| 80K x 4096 | 90.0% | 30.319 | 0.190 | 9.353 | 39.851 |
| 10K x 4096 | 99.5% | 3.730 | 0.047 | 1.094 | 4.860 |
| 20K x 4096 | 99.5% | 7.597 | 0.069 | 2.168 | 9.824 |
| 40K x 4096 | 99.5% | 13.576 | 0.107 | 4.310 | 17.986 |
| 80K x 4096 | 99.5% | 30.042 | 0.189 | 8.590 | 38.811 |

H2D and Build BaSC overlap across four CUDA streams. GPU Total is measured on the critical path and is not the arithmetic sum of the separately reported phase values.

The command also reports `Device-side Work`, defined as Build BaSC plus BaSC-GEMM, so that the output maps directly to the H2D, Device-side Work, and Total columns used in the paper. Exact paper values require the original datasets; the built-in benchmark uses random binary input with the requested dimensions and sparsity.
