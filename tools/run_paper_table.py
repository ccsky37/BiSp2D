#!/usr/bin/env python3

import csv
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
binary = root / "build" / "bisp2d"
datasets = root / "datasets" / "paper_datasets.csv"
warmups = sys.argv[1] if len(sys.argv) > 1 else "5"
runs = sys.argv[2] if len(sys.argv) > 2 else "5"


def read_value(output, name):
    prefix = name + ": "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].split()[0]


rows = []
with datasets.open(newline="") as source:
    for dataset in csv.DictReader(source):
        command = [
            str(binary),
            dataset["instances"],
            dataset["features"],
            dataset["sparsity_percent"],
            "0",
            warmups,
            runs,
            "native",
        ]
        print("Running " + dataset["dataset"] + "...", file=sys.stderr)
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        rows.append([
            dataset["dataset"],
            f'{int(dataset["instances"]):,}',
            f'{int(dataset["features"]):,}',
            dataset["sparsity_percent"] + "%",
            read_value(output, "H2D"),
            read_value(output, "Device-side Work"),
            read_value(output, "GPU Total"),
        ])

print("| Dataset | Instances | Features | Sparsity | H2D (ms) | Device-side Work (ms) | Total (ms) |")
print("|---|---:|---:|---:|---:|---:|---:|")
for row in rows:
    print("| " + " | ".join(row) + " |")
