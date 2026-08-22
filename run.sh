#!/usr/bin/env bash
set -e

make -j
./build/bisp2d "$@"
