#!/usr/bin/env bash
# Data for the leakage-impact experiments. Everything comes from public GitHub repositories.
#   CIFAR-10 + ciFAIR lists : see ../download_cifar.sh (run it with the same OUT)
#   UCI HAR                  : mirror greenglobal/uci-har-dataset (counts checked in e2: 7,352 / 2,947 windows, 21 / 9 subjects)
#   ESC-50                   : karolpiczak/ESC-50 (official)
#   AG News                  : mhjabreel/CharCnn_Keras (data/ag_news_csv)
#   TREC-6, SST-2, SST-5     : harvardnlp/sent-conv-torch (data/)
set -euo pipefail
OUT=${1:-.bench}
mkdir -p "$OUT"
bash "$(dirname "$0")/../download_cifar.sh" "$OUT/cifar10"
git clone -q --depth 1 https://github.com/greenglobal/uci-har-dataset.git "$OUT/har"
git clone -q --depth 1 https://github.com/karolpiczak/ESC-50.git "$OUT/esc50"
git clone -q --depth 1 https://github.com/mhjabreel/CharCnn_Keras.git "$OUT/agnews"
git clone -q --depth 1 https://github.com/harvardnlp/sent-conv-torch.git "$OUT/sentconv"
echo "export BENCH_DATA=$OUT"
