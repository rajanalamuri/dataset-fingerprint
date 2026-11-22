"""Reproduce the ciFAIR finding (Barz & Denzler 2020) with dataset-fingerprint's near-duplicate check.

ciFAIR found 286 CIFAR-10 test images with a near-duplicate in the training set, using CNN
features plus manual review. Here a plain 64-bit dHash + exact Hamming search runs inside
`commit`, and we measure how many of those 286 it recovers at each threshold, and how many
other test images it flags.

    bash benchmarks/download_cifar.sh .bench/cifar10
    python benchmarks/run_cifar.py --root .bench/cifar10 --work .bench/cifar-work
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

from PIL import Image

import dataset_fingerprint as fp
from dataset_fingerprint import perceptual

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SHA256 = {
    "data_batch_1.bin": "cee916563c9f80d84e3cc88e17fdc0941787f1244f00a67874d45b261883ada5",
    "data_batch_2.bin": "a591ca11fa1708a91ee40f54b3da4784ccd871ecf2137de63f51ada8b3fa57ed",
    "data_batch_3.bin": "bbe8596564c0f86427f876058170b84dac6670ddf06d79402899d93ceea26f67",
    "data_batch_4.bin": "014e562d6e23c72197cc727519169a60359f5eccd8945ad5a09d710285ff4e48",
    "data_batch_5.bin": "755304fc0b379caeae8c14f0dac912fbc7d6cd469eb67a1029a08a39453a9add",
    "test_batch.bin": "8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b",
}
THRESHOLDS = (0, 2, 4, 6, 8, 10)


def materialise(root: Path, out: Path) -> None:
    """CIFAR .bin batches -> train/<class>/<id>.png, test/<class>/<id>.png (lossless)."""
    if (out / ".done").exists():
        return
    shutil.rmtree(out, ignore_errors=True)
    for name, digest in SHA256.items():
        got = hashlib.sha256((root / "bin" / name).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{name}: sha256 {got} != pinned {digest}")
    jobs = [(f"data_batch_{i}.bin", "train", (i - 1) * 10000) for i in range(1, 6)] + [("test_batch.bin", "test", 0)]
    for name, split, offset in jobs:
        raw = (root / "bin" / name).read_bytes()
        for k in range(10000):
            rec = raw[k * 3073 : (k + 1) * 3073]
            label, px = rec[0], rec[1:]
            rgb = bytes(b for i in range(1024) for b in (px[i], px[1024 + i], px[2048 + i]))
            d = out / split / CLASSES[label]
            d.mkdir(parents=True, exist_ok=True)
            Image.frombytes("RGB", (32, 32), rgb).save(d / f"{offset + k:05d}.png", optimize=False)
    (out / ".done").write_text("ok")


def idx(path: str) -> int:
    return int(path.rsplit("/", 1)[-1].split(".")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "cifar.json"))
    a = ap.parse_args()
    root, work = Path(a.root), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    os.environ["FINGERPRINT_CACHE"] = str(work / "cache.sqlite")
    if os.path.exists(os.environ["FINGERPRINT_CACHE"]):
        os.remove(os.environ["FINGERPRINT_CACHE"])

    t0 = time.time()
    data = work / "cifar10"
    materialise(root, data)
    t_mat = time.time() - t0

    listed = list(csv.DictReader(open(root / "cifair" / "duplicates_cifar10.csv")))
    cifair = {int(r["TestID"]): (int(r["TrainID"]), int(r["Judgment"])) for r in listed}
    judg = Counter(j for _, j in cifair.values())

    # 1. plain commit (no near-dup check) vs commit with it - same files, cold hash caches
    shutil.rmtree(work / "ledger-plain", ignore_errors=True)
    t = time.time()
    plain = fp.Ledger(str(work / "ledger-plain")).commit(str(data), message="CIFAR-10", quiet=True, cache=False)
    t_plain = time.time() - t
    shutil.rmtree(work / "ledger-nd", ignore_errors=True)
    t = time.time()
    v = fp.Ledger(str(work / "ledger-nd")).commit(str(data), message="CIFAR-10 + near-dups", near_duplicates=4, quiet=True)
    t_nd = time.time() - t
    t = time.time()
    v_warm = fp.Ledger(str(work / "ledger-nd")).commit(str(data), message="re-commit", near_duplicates=4, quiet=True, allow_empty=True)
    t_warm = time.time() - t

    q = v.manifest["quality"]
    recs = [r for r in v.records() if (r.get("meta") or {}).get("dhash")]
    items = [(r["split"], int(r["meta"]["dhash"], 16)) for r in recs]
    paths = [r["source_path"] for r in recs]

    sweep = []
    for T in THRESHOLDS:
        t = time.time()
        pairs, hot = perceptual.cross_split_pairs(items, T)
        dt = time.time() - t
        match: dict = {}
        for i, j, _d in pairs:
            si, sj = items[i][0], items[j][0]
            if {si, sj} != {"train", "test"}:
                continue
            te, tr = (i, j) if si == "test" else (j, i)
            match.setdefault(idx(paths[te]), set()).add(idx(paths[tr]))
        flagged = set(match)
        hit = flagged & set(cifair)
        same_pair = sum(1 for k in hit if cifair[k][0] in match[k])
        by_j = {f"judgment_{jj}": sum(1 for k in hit if cifair[k][1] == jj) for jj in sorted(judg)}
        sweep.append(
            {
                "threshold": T,
                "test_images_flagged": len(flagged),
                "in_cifair_list": len(hit),
                "precision_vs_cifair": round(len(hit) / len(flagged), 3) if flagged else None,
                "recall_vs_cifair": round(len(hit) / len(cifair), 3),
                "matched_same_train_image": same_pair,
                "recovered_by_judgment": by_j,
                "search_seconds": round(dt, 2),
                "skipped_hot_bucket_items": hot,
            }
        )
        print(sweep[-1])

    unlisted = []
    p4, _ = perceptual.cross_split_pairs(items, 4)
    for i, j, d in sorted(p4, key=lambda p: p[2]):
        te, tr = (i, j) if items[i][0] == "test" else (j, i)
        if idx(paths[te]) not in cifair:
            unlisted.append([paths[te], paths[tr], d])

    res = {
        "dataset": "CIFAR-10 (50,000 train / 10,000 test, PNG)",
        "reference": "ciFAIR duplicates_cifar10.csv (Barz & Denzler 2020)",
        "cifair_listed": len(cifair),
        "cifair_by_judgment": {str(k): v for k, v in sorted(judg.items())},
        "files": dict(Counter(r["split"] for r in plain.records())),
        "exact_duplicates_across_splits": q["duplicates_across_splits"],
        "manifest_near_duplicates_T4": {k: v for k, v in q["near_duplicates"].items() if k != "examples"},
        "sweep": sweep,
        "unlisted_T4_examples": unlisted[:40],
        "timings_sec": {
            "materialise_png": round(t_mat, 1),
            "commit_plain_cold": round(t_plain, 1),
            "commit_with_near_dups_cold": round(t_nd, 1),
            "recommit_with_near_dups_warm": round(t_warm, 1),
        },
        "unchanged_recommit_same_fingerprint": v_warm.fingerprint == v.fingerprint,
        "env": {"python": platform.python_version(), "machine": platform.machine(), "dataset-fingerprint": fp.__version__, "cpus": os.cpu_count()},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("exact_duplicates_across_splits", "manifest_near_duplicates_T4", "timings_sec")}, indent=2))


if __name__ == "__main__":
    main()
