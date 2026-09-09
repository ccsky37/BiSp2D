# Comparison data

These are archived comparison measurements used in the BiSp2D revision. They are not new benchmark runs of the current public checkout. Dataset names identify the recorded experiment configurations; this directory does not contain the input matrices.

## Sources

| File | Source in the experiment repository | Role |
|---|---|---|
| [a100_raw.csv](a100_raw.csv) | `experiments/a100_baseline/results_best_kernel_20260530/a100_baseline_raw.csv` | 180 rows: 9 datasets × 4 methods × 5 measured runs. |
| [a100_summary.csv](a100_summary.csv) | Computed from the included A100 raw CSV. | Arithmetic means and sample standard deviations. |
| [p100_summary.csv](p100_summary.csv) | `experiments/p100_baseline/results_final_baseline/p100_final_baseline_summary.csv` | 9 summaries: 3 configurations × 3 methods, 10 measured runs each. |

A100 raw records retain dimensions, target sparsity, actual sparsity, and per-run timings. Target and actual sparsity can differ; use the recorded actual values when interpreting these results. The P100 source summary retains dimensions but does not include a sparsity field. Do not substitute the settings in the current public `datasets.csv` for missing metadata.

P100 configurations are `200000 × 256`, `37728 × 320`, and `50000 × 2048`. Sputnik denotes the integer-adapted baseline, not an unmodified upstream FP32 build. The P100 plot uses the archived final summaries; it does not reconstruct individual samples.

## Timing

- **H2D:** uncompressed input transfer.
- **Device-side work:** GPU format construction/conversion plus computation.
- **Total:** independently measured H2D-to-device-completion latency, taken from `gpu_total_ms` (A100) or `total_mean` (P100).
- **Error bars:** sample standard deviation of total latency, not a confidence interval and not the sum of phase standard deviations.
- **Speedup:** baseline mean total / BiSp2D mean total for the same dataset configuration and GPU.

H2D transfer and BaSC construction can overlap. Total latency is not reconstructed by adding separately measured phase means.

## Regenerate figures

From the repository root:

```bash
python3 -m pip install matplotlib
python3 docs/plot_results.py
```

This rebuilds `a100_summary.csv` and the two PNGs under `docs/images/`. No GPU is required. It redraws existing measurements; it does not rerun baseline kernels.
