from pathlib import Path
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
COLORS = {"BiSp2D": "#9BC7BB", "cuSPARSE": "#EDCE82", "Sputnik": "#88A6CD",
          "cuBLASLt": "#C4C9D0", "CUTLASS": "#EAA18A"}
NAMES = {"cuBLASLt INT8 baseline": "cuBLASLt", "CUTLASS INT8 GEMM baseline": "CUTLASS",
         "cuSPARSE SpMM baseline": "cuSPARSE", "BiSp2D baseline": "BiSp2D"}


def read_results():
    groups = {}
    with (ROOT / "results/a100_raw.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = row["dataset"], NAMES[row["method"]]
            groups.setdefault(key, []).append(row)
    a100 = []
    for (dataset, method), rows in groups.items():
        assert len(rows) == 5
        metrics = {
            "h2d": [float(r["h2d_ms"]) for r in rows],
            "kernel": [float(r["build_kernel_ms"]) + float(r["compute_kernel_ms"]) for r in rows],
            "total": [float(r["gpu_total_ms"]) for r in rows],
        }
        out = {"dataset": dataset, "method": method, "runs": len(rows)}
        for key, values in metrics.items():
            out[key + "_mean"] = statistics.mean(values)
            out[key + "_std"] = statistics.stdev(values)
        a100.append(out)
    fields = list(a100[0])
    with (ROOT / "results/a100_summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(a100)
    with (ROOT / "results/p100_summary.csv").open(newline="") as source:
        p100 = list(csv.DictReader(source))
    return a100, p100


def draw(data, methods, platform, output, ncols, nrows, titles, subtitle):
    datasets = list(dict.fromkeys(row["dataset"] for row in data))
    assert len(datasets) == ncols * nrows
    lookup = {(r["dataset"], r["method"]): r for r in data}
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows + 0.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(platform if not subtitle else platform + " | Dataset comparison", x=0.035, y=0.98,
                 ha="left", fontsize=17, fontweight="bold")
    if subtitle:
        fig.text(0.035, 0.86 if nrows == 1 else 0.95, subtitle, fontsize=10, color="#555555")
    for ax, dataset in zip(axes.flat, datasets):
        rows = [lookup[dataset, method] for method in methods]
        means = [float(r["total_mean"]) for r in rows]
        errors = [float(r["total_std"]) for r in rows]
        limit = max(v + e for v, e in zip(means, errors))
        ax.barh(methods, means, xerr=errors, height=0.58,
                color=[COLORS[m] for m in methods], edgecolor="#333333", linewidth=0.8,
                error_kw={"ecolor": "#454545", "elinewidth": 0.75, "capsize": 2})
        ax.invert_yaxis()
        ax.set_xlim(0, limit * 1.3)
        ax.set_title(titles.get(dataset, dataset), loc="left", fontsize=11, pad=12)
        ax.set_xlabel("Total time (ms)", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E7E9EB", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#8B9096")
        for i, (value, error) in enumerate(zip(means, errors)):
            ax.text(value + error + limit * 0.025, i, f"{value:.2f}", va="center", fontsize=9)
    top = (0.81 if nrows == 1 else 0.925) if subtitle else 0.955
    fig.tight_layout(rect=(0.015, 0.01, 0.995, top), h_pad=2.8, w_pad=2)
    fig.savefig(ROOT / "images" / output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    a100, p100 = read_results()
    draw(a100, ["BiSp2D", "cuSPARSE", "cuBLASLt", "CUTLASS"], "A100",
         "a100_comparison.png", 3, 3,
         {"Beauty & Personal Care": "Beauty & Personal Care",
          "Clothing, Shoes & Jewelry": "Clothing, Shoes & Jewelry",
          "Beauty & Personal Care (meta)": "Beauty & Personal Care (meta)",
          "Amazon Subscription Boxes": "Amazon Subscription Boxes"},
         "")
    draw(p100, ["BiSp2D", "cuSPARSE", "Sputnik"], "Tesla P100",
         "p100_comparison.png", 3, 1,
         {"Criteo256": "Criteo | 200,000 x 256", "MSWeb320": "MSWeb | 37,728 x 320",
          "MovieLens2048": "MovieLens | 50,000 x 2,048"},
         "Total latency, including H2D | Mean and standard deviation, 10 runs | Lower is better | Independent panel scales")
