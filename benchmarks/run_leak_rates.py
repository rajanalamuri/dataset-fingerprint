"""Leak rate and effective test size for every benchmark dataset.

Runs after the other benchmark scripts, on the folders they prepared:

    python benchmarks/run_leak_rates.py --mnist work/mnist/data --fashion work/fashion/data \
        --cifar .bench/cifar-work/cifar10 --coco128 .bench/coco-work/coco128

MNIST/Fashion folders also hold 600 flipped train images from run_benchmark.py; the "as published" runs
exclude them. The "injected" runs mirror the folder (hard links) and plant known leaks: 50 test images
copied into train under new names and 50 horizontally flipped test images saved into train as
<name>_hflip.png - so the true leak rate is exactly 100 / 10,000 = 1.00%.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import dataset_fingerprint as fp


def run(name: str, src: str, **kw) -> dict:
    work = Path(tempfile.mkdtemp(prefix="leak-"))
    os.environ["FINGERPRINT_CACHE"] = str(work / "cache.sqlite")
    t = time.time()
    v = fp.Ledger(str(work / "ledger")).commit(src, message=name, quiet=True, **kw)
    dt = time.time() - t
    q = v.manifest["quality"]
    lk = q["leakage"]
    nd = q.get("near_duplicates") or {}
    shutil.rmtree(work, ignore_errors=True)
    out = {
        "dataset": name,
        "leak_rate": lk.get("leak_rate"),
        "leaked_files": lk.get("leaked_files"),
        "evaluation_files": lk.get("evaluation_files"),
        "suspected_rate": lk.get("suspected_rate"),
        "suspected_files": lk.get("suspected_files"),
        "near_duplicates_uninformative": lk.get("near_duplicates_uninformative"),
        "by_split": lk.get("by_split"),
        "recording_names_also_in_train": lk.get("recording_names_also_in_train"),
        "near_duplicate_examples": nd.get("examples", [])[:10],
        "commit_seconds": round(dt, 1),
        "options": {k: v for k, v in kw.items() if k not in ("class_names",)},
    }
    print(json.dumps({k: out[k] for k in ("dataset", "leak_rate", "leaked_files", "suspected_files", "near_duplicates_uninformative")}))
    return out


def inject(src: str, n: int = 50) -> str:
    from PIL import Image, ImageOps

    dst = Path(tempfile.mkdtemp(prefix="inject-")) / "data"
    shutil.copytree(src, dst, copy_function=os.link, ignore=shutil.ignore_patterns("*_hflip*"))
    tests = sorted((dst / "test").rglob("*.png"))
    step = len(tests) // (2 * n)
    picks = tests[::step][: 2 * n]
    for k, p in enumerate(picks):
        target = dst / "train" / p.parent.name
        if k < n:  # exact copy, new name
            shutil.copyfile(p, target / f"copied_{k:03d}.png")
        else:  # augmented copy of a test image, named after its original
            ImageOps.mirror(Image.open(p)).save(target / f"{p.stem}_hflip.png")
    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist")
    ap.add_argument("--fashion")
    ap.add_argument("--cifar")
    ap.add_argument("--coco128")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "leak_rates.json"))
    a = ap.parse_args()
    res = []
    for name, path in (("MNIST", a.mnist), ("Fashion-MNIST", a.fashion)):
        if path:
            res.append(run(f"{name} (as published)", path, exclude=["*_hflip*"], near_duplicates=4))
            inj = inject(path)
            r = run(f"{name} + 100 planted leaks (50 copies, 50 flips)", inj)
            r["planted"] = {"exact_copy": 50, "augmented_copy": 50, "true_leak_rate": 0.01}
            res.append(r)
            shutil.rmtree(Path(inj).parent, ignore_errors=True)
    if a.cifar:
        res.append(run("CIFAR-10 (as published)", a.cifar, near_duplicates=4))
    if a.coco128:
        res.append(run("coco128 (hash split 80/10/10)", a.coco128, split="0.8,0.1,0.1", label_format="yolo", near_duplicates=4))
    Path(a.out).write_text(
        json.dumps(
            {"results": res, "env": {"python": platform.python_version(), "cpus": os.cpu_count(), "dataset-fingerprint": fp.__version__}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
