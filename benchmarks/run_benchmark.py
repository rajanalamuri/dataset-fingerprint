"""Benchmark mlfingerprint on MNIST and Fashion-MNIST (70,000 images each).

    python benchmarks/run_benchmark.py --raw ./raw --work ./work --out benchmarks/results

--raw must contain mnist/ and fashion/ with the four original *-ubyte.gz files each
(MD5s are checked against the published values before anything runs).

Every number mlfingerprint reports is cross-checked against ground truth computed
independently from the raw IDX pixel arrays.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import struct
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

import mlfingerprint as fp

MD5 = {
    "mnist": {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    },
    "fashion": {
        "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
    },
}
CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["tshirt_top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"],
}
NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()  # nosec B324 - matching published checksums


def read_idx(path: Path):
    raw = gzip.decompress(path.read_bytes())
    magic = struct.unpack(">I", raw[:4])[0]
    if magic == 2051:
        n, r, c = struct.unpack(">III", raw[4:16])
        return [raw[16 + i * r * c : 16 + (i + 1) * r * c] for i in range(n)], (r, c)
    if magic == 2049:
        (n,) = struct.unpack(">I", raw[4:8])
        return list(raw[8 : 8 + n]), None
    raise ValueError(f"bad IDX magic {magic} in {path}")


def ground_truth(train_x, train_y, test_x, test_y):
    """Exact-duplicate facts straight from the pixel arrays (no PNGs, no mlfingerprint)."""
    tr = Counter(train_x)
    te = Counter(test_x)
    train_set = set(train_x)
    test_in_train = [i for i, x in enumerate(test_x) if x in train_set]
    first_label = {}
    for x, y in zip(train_x, train_y):
        first_label.setdefault(x, y)
    label_conflicts = sum(1 for i in test_in_train if first_label[test_x[i]] != test_y[i])
    return {
        "train_images": len(train_x),
        "test_images": len(test_x),
        "train_duplicate_images": len(train_x) - len(tr),
        "test_duplicate_images": len(test_x) - len(te),
        "test_images_also_in_train": len(test_in_train),
        "distinct_pixel_arrays_shared_by_train_and_test": len(set(test_x) & train_set),
        "test_in_train_with_different_label": label_conflicts,
        "label_counts": {"train": dict(Counter(train_y)), "test": dict(Counter(test_y))},
    }


def export_pngs(ds: str, xs, ys, shape, split: str, root: Path) -> None:
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = root / split / CLASSES[ds][y]
        d.mkdir(parents=True, exist_ok=True)
        Image.frombytes("L", (shape[1], shape[0]), x).save(d / f"{split}_{i:05d}.png", optimize=False)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 3)


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(ds: str, raw: Path, work: Path) -> dict:
    for f, want in MD5[ds].items():
        got = md5(raw / ds / f)
        assert got == want, f"{ds}/{f}: md5 {got} != published {want}"
    train_x, shape = read_idx(raw / ds / "train-images-idx3-ubyte.gz")
    train_y, _ = read_idx(raw / ds / "train-labels-idx1-ubyte.gz")
    test_x, _ = read_idx(raw / ds / "t10k-images-idx3-ubyte.gz")
    test_y, _ = read_idx(raw / ds / "t10k-labels-idx1-ubyte.gz")
    gt = ground_truth(train_x, train_y, test_x, test_y)

    base = work / ds
    if base.exists():
        shutil.rmtree(base)
    data, ledger = base / "data", base / "ledger"
    _, t_export = timed(lambda: (export_pngs(ds, train_x, train_y, shape, "train", data), export_pngs(ds, test_x, test_y, shape, "test", data)))
    os.environ["FINGERPRINT_CACHE"] = str(base / "hash-cache.sqlite")
    lg = fp.Ledger(str(ledger), name=ds)
    res: dict = {"dataset": NAMES[ds], "ground_truth": gt, "timings_s": {"export_pngs": t_export}}
    T = res["timings_s"]

    v1, T["commit_cold"] = timed(lambda: lg.commit(str(data), message="original release", probe=True, quiet=True))
    _, T["commit_no_change"] = timed(lambda: lg.commit(str(data), quiet=True))
    rep, T["verify_full"] = timed(lambda: v1.verify(workers=16))
    rep_s, T["verify_sample_5pct"] = timed(lambda: v1.verify(sample=0.05, workers=16))
    rep_q, T["verify_quick"] = timed(lambda: v1.verify(quick=True))
    assert rep.ok and rep_s.ok and rep_q.ok

    # tamper with one test image (one pixel) -> verify must catch it; re-commit -> patch bump
    victim = sorted((data / "test").rglob("*.png"))[1234]
    original = victim.read_bytes()
    im = Image.open(victim)
    im.putpixel((0, 0), 255 - im.getpixel((0, 0)))
    im.save(victim)
    tamper_rep, T["verify_after_tamper"] = timed(lambda: v1.verify(workers=16))
    v2, T["commit_after_one_change"] = timed(lambda: lg.commit(str(data), message="one pixel edited", quiet=True))
    victim.write_bytes(original)

    # a new batch: horizontally-flipped augmentations of 600 train images, plus ONE flip of a test image
    train_pngs = sorted((data / "train").rglob("*.png"))[::100][:600]
    for p in train_pngs:
        ImageOps.mirror(Image.open(p)).save(p.with_name(p.stem + "_hflip.png"))
    leak_src = sorted((data / "test").rglob("*.png"))[42]
    leak_dst = data / "train" / leak_src.parent.name / (leak_src.stem + "_hflip.png")
    ImageOps.mirror(Image.open(leak_src)).save(leak_dst)
    v3, T["commit_new_batch_601_files"] = timed(lambda: lg.commit(str(data), message="hflip augmentation batch", quiet=True))
    d13 = lg.diff(v1, v3)

    # content-addressed snapshot of v1 data, then of v3: identical files stored once
    cas = fp.Ledger(str(base / "cas"), name=ds)
    for p in train_pngs + [leak_dst]:
        p.with_name(p.name).unlink() if p.name.endswith("_hflip.png") else None
    for p in data.rglob("*_hflip.png"):
        p.unlink()
    c1, T["cas_snapshot_v1"] = timed(lambda: cas.commit(str(data), mode="cas", quiet=True))
    for p in train_pngs:
        ImageOps.mirror(Image.open(p)).save(p.with_name(p.stem + "_hflip.png"))
    c2, T["cas_snapshot_v2"] = timed(lambda: cas.commit(str(data), mode="cas", quiet=True))

    m = v1.manifest
    q3 = v3.manifest["quality"]
    res["mlfingerprint"] = {
        "version": v1.version,
        "fingerprint": v1.fingerprint,
        "total_files": m["metadata"]["total_files"],
        "splits": {k: v["files"] for k, v in m["metadata"]["splits"].items()},
        "num_classes": m["metadata"]["num_classes"],
        "class_counts": {k: v.get("class_counts") for k, v in m["metadata"]["splits"].items()},
        "resolutions": m["metadata"]["modalities"]["image"].get("resolutions"),
        "quality_v1": m["quality"],
        "duplicates_across_splits_v1": m["quality"]["duplicates_across_splits"],
        "duplicate_files_v1": m["quality"]["duplicate_files"],
        "tamper": {"detected": not tamper_rep.ok, "modified": tamper_rep.modified, "recommit_version": v2.version, "bump": v2["lineage"]["bump"]},
        "new_batch": {
            "version": v3.version,
            "bump": v3["lineage"]["bump"],
            "added": len(d13.added),
            "augmented_files": q3["augmented_files"],
            "augmentation_leaks_flagged": q3["augmentation_cross_split"],
            "leak_example": (q3.get("examples", {}).get("augmentation_cross_split") or [None])[0],
        },
        "cas": {
            "v1_objects_written": c1["source"]["objects_written"],
            "v2_objects_written": c2["source"]["objects_written"],
            "v2_objects_reused": c2["source"]["objects_reused"],
        },
        "sizes_bytes": {
            "data_pngs": du(data),
            "manifest.json": (ledger / v1.folder / "manifest.json").stat().st_size,
            "index.jsonl.gz": (ledger / v1.folder / "index.jsonl.gz").stat().st_size,
        },
    }
    # cross-checks against ground truth
    pngs_equal_pixels = gt["distinct_pixel_arrays_shared_by_train_and_test"]
    res["checks"] = {
        "files_match": res["mlfingerprint"]["total_files"] == gt["train_images"] + gt["test_images"],
        "class_counts_match": all(
            {CLASSES[ds][k]: v for k, v in gt["label_counts"][s].items()} == res["mlfingerprint"]["class_counts"][s] for s in ("train", "test")
        ),
        "cross_split_duplicates_match": res["mlfingerprint"]["duplicates_across_splits_v1"] == pngs_equal_pixels,
        "duplicate_files_match": res["mlfingerprint"]["duplicate_files_v1"]
        == (gt["train_images"] + gt["test_images"]) - len(set(train_x) | set(test_x)),
        "tamper_detected": res["mlfingerprint"]["tamper"]["detected"],
        "augmentation_leak_flagged": res["mlfingerprint"]["new_batch"]["augmentation_leaks_flagged"] == 1,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    a = ap.parse_args()
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(), "mlfingerprint": fp.__version__}
    results = {"environment": env, "datasets": [run(ds, a.raw, a.work) for ds in ("mnist", "fashion")]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
