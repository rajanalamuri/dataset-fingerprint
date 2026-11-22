#!/usr/bin/env bash
# CIFAR-10 binary batches + the ciFAIR duplicate lists (Barz & Denzler 2020).
# Preferred: the official https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (extract cifar-10-batches-bin/).
# Fallback used for the published numbers: the same .bin files committed in github.com/quietlychris/cifar-ten
# (commit c20b366); run_cifar.py pins their sha256; its test batch was cross-checked against ciFAIR-10
# (all 9,714 images ciFAIR kept are pixel-identical; all 286 it replaced differ).
set -euo pipefail
OUT=${1:-./raw/cifar10}
mkdir -p "$OUT/bin" "$OUT/cifair"
if [ ! -f "$OUT/bin/test_batch.bin" ]; then
  tmp=$(mktemp -d)
  git clone -q --filter=blob:none --no-checkout https://github.com/quietlychris/cifar-ten.git "$tmp/r"
  for f in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch; do
    git -C "$tmp/r" show "c20b366:data/cifar-10-batches-bin/$f.bin" > "$OUT/bin/$f.bin"
  done
  rm -rf "$tmp"
fi
base=https://raw.githubusercontent.com/cvjena/cifair/master
curl -fsSL -o "$OUT/cifair/duplicates_cifar10.csv" "$base/meta/duplicates_cifar10.csv"
curl -fsSL -o "$OUT/cifair/duplicates_cifar10_test.csv" "$base/meta/duplicates_cifar10_test.csv"
echo "downloaded to $OUT"
