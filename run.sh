#!/usr/bin/env bash
set -e

make -j
"${PYTHON:-python3}" bench.py "$@"
