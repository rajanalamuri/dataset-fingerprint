#!/usr/bin/env bash
# COCO 2017 labels (all train2017 + val2017 annotations, YOLO format + official instances_val2017.json)
# and coco128, as redistributed by Ultralytics on GitHub releases.
set -euo pipefail
OUT=${1:-./coco}
mkdir -p "$OUT"
base=https://github.com/ultralytics/assets/releases/download/v0.0.0
curl -fsSL -o "$OUT/coco2017labels.zip" "$base/coco2017labels.zip"
curl -fsSL -o "$OUT/coco128.zip" "$base/coco128.zip"
python3 -I -c "import sys, zipfile; [zipfile.ZipFile(z).extractall(sys.argv[1]) for z in sys.argv[2:]]" \
  "$OUT" "$OUT/coco2017labels.zip" "$OUT/coco128.zip"
echo "extracted to $OUT (coco/, coco128/)"
