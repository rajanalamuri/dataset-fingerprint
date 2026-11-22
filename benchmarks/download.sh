#!/usr/bin/env bash
# Fetch the original MNIST and Fashion-MNIST files (MD5s are checked by run_benchmark.py).
set -euo pipefail
OUT=${1:-./raw}
mkdir -p "$OUT/mnist" "$OUT/fashion"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fsSL -o "$OUT/mnist/$f.gz"   "https://raw.githubusercontent.com/fgnt/mnist/master/$f.gz"
  curl -fsSL -o "$OUT/fashion/$f.gz" "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f.gz"
done
echo "downloaded to $OUT"
