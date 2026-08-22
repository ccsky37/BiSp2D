#!/usr/bin/env bash
set -e

make -j
"${PYTHON:-python3}" tools/run_paper_table.py "$@"
