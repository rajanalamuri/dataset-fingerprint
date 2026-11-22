"""Benchmark mlfingerprint on COCO 2017.

Part A - the full COCO 2017 annotation set (122,218 YOLO label files for train2017 + val2017,
         plus the official instances_val2017.json), as redistributed by Ultralytics.
Part B - coco128: 128 real COCO train2017 images with their labels (image <-> label pairing).

    bash benchmarks/download_coco.sh .bench/coco
    python benchmarks/run_coco.py --root .bench/coco --work .bench/coco-work

The COCO image pixels themselves (19 GB) are not part of this run; see RESULTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

from PIL import Image

import mlfingerprint as fp

COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 3)


def yolo_counts(folder: Path) -> Counter:
    c: Counter = Counter()
    for p in folder.glob("*.txt"):
        for line in p.read_text().splitlines():
            parts = line.split()
            if parts:
                c[COCO80[int(parts[0])]] += 1
    return c


def part_a(root: Path, work: Path) -> dict:
    src = work / "coco-annotations"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(root / "coco", src)
    lab = src / "labels"
    # ---- ground truth, computed without mlfingerprint
    gt_files = {s: len(list((lab / s).glob("*.txt"))) for s in ("train2017", "val2017")}
    gt_objects = {s: yolo_counts(lab / s) for s in ("train2017", "val2017")}
    official = json.loads((src / "annotations" / "instances_val2017.json").read_text())
    cat_index = {c["id"]: i for i, c in enumerate(sorted(official["categories"], key=lambda c: c["id"]))}
    off_noncrowd: Counter = Counter()
    crowd = 0
    for a in official["annotations"]:
        if a.get("iscrowd"):
            crowd += 1
        else:
            off_noncrowd[COCO80[cat_index[a["category_id"]]]] += 1
    imgs_listed = {s: sum(1 for _ in open(src / f"{s}.txt")) for s in ("train2017", "val2017")}
    hashes: dict = {}
    for s in ("train2017", "val2017"):
        for p in (lab / s).glob("*.txt"):
            hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(f"{s}/{p.name}")
    dup_groups = [sorted(v) for v in hashes.values() if len(v) > 1]
    gt_dup_files = sum(len(v) - 1 for v in dup_groups)
    gt_cross = sum(1 for v in dup_groups if len({x.split("/")[0] for x in v}) > 1)

    # ---- mlfingerprint
    os.environ["FINGERPRINT_CACHE"] = str(work / "cache-a.sqlite")
    lg = fp.Ledger(str(work / "ledger-a"), name="coco2017-annotations")
    T: dict = {}
    v1, T["commit_cold"] = timed(lambda: lg.commit(str(src), message="COCO 2017 labels", label_format="yolo", class_names=COCO80, quiet=True))
    _, T["commit_no_change"] = timed(lambda: lg.commit(str(src), label_format="yolo", class_names=COCO80, quiet=True))
    full, T["verify_full"] = timed(lambda: v1.verify(workers=16))
    smp, T["verify_sample_5pct"] = timed(lambda: v1.verify(sample=0.05, workers=16))
    qk, T["verify_quick"] = timed(lambda: v1.verify(quick=True))
    assert full.ok and smp.ok and qk.ok

    victim = sorted((lab / "val2017").glob("*.txt"))[100]
    orig = victim.read_text()
    victim.write_text(orig.replace("0.", "0.0", 1))  # nudge one box coordinate
    tamper, T["verify_after_tamper"] = timed(lambda: v1.verify(workers=16))
    v2, T["commit_after_one_change"] = timed(
        lambda: lg.commit(str(src), message="one box edited", label_format="yolo", class_names=COCO80, quiet=True)
    )
    victim.write_text(orig)
    # leak: a val annotation copied into train under a new name
    leak = lab / "train2017" / "999999999999.txt"
    leak.write_bytes(sorted((lab / "val2017").glob("*.txt"))[7].read_bytes())
    v3, T["commit_with_leak"] = timed(lambda: lg.commit(str(src), message="leak test", label_format="yolo", class_names=COCO80, quiet=True))
    leak.unlink()

    m = v1.manifest
    split_files = {k: v["files"] for k, v in m["metadata"]["splits"].items()}
    cc = {k: v.get("class_counts", {}) for k, v in m["metadata"]["splits"].items()}
    path_split = {r["source_path"]: r["split"] for r in v1.records()}
    res = {
        "dataset": "COCO 2017 annotations (YOLO format, Ultralytics redistribution)",
        "ground_truth": {
            "label_files": gt_files,
            "images_listed": imgs_listed,
            "images_without_labels": {s: imgs_listed[s] - gt_files[s] for s in gt_files},
            "objects": {s: sum(c.values()) for s, c in gt_objects.items()},
            "official_val_annotations": len(official["annotations"]),
            "official_val_crowd": crowd,
            "official_val_noncrowd": sum(off_noncrowd.values()),
            "identical_label_file_groups": dup_groups,
            "extra_identical_files": gt_dup_files,
            "cross_split_identical_groups": gt_cross,
            "top5_train": gt_objects["train2017"].most_common(5),
        },
        "timings_s": T,
        "mlfingerprint": {
            "version": v1.version,
            "fingerprint": v1.fingerprint,
            "total_files": m["metadata"]["total_files"],
            "split_files": split_files,
            "num_classes": m["metadata"].get("num_classes"),
            "label_source": m["metadata"].get("label_source"),
            "annotation_json_split": path_split.get("annotations/instances_val2017.json"),
            "list_file_splits": {f: path_split.get(f) for f in ("train2017.txt", "val2017.txt", "test-dev2017.txt")},
            "quality": {k: v for k, v in m["quality"].items() if k != "examples"},
            "tamper": {"detected": not tamper.ok, "modified": tamper.modified, "recommit": v2.version, "bump": v2["lineage"]["bump"]},
            "leak": {
                "version": v3.version,
                "duplicates_across_splits": v3["quality"]["duplicates_across_splits"],
                "warning": next((w for w in v3["lineage"]["warnings"] if "identical" in w), None),
            },
            "sizes_bytes": {
                "data": sum(p.stat().st_size for p in src.rglob("*") if p.is_file()),
                "manifest.json": (work / "ledger-a" / v1.folder / "manifest.json").stat().st_size,
                "index.jsonl.gz": (work / "ledger-a" / v1.folder / "index.jsonl.gz").stat().st_size,
            },
        },
    }
    res["checks"] = {
        "label_files_per_split_match": split_files.get("train", 0) - 1 == gt_files["train2017"]
        and split_files.get("val", 0) - 2 == gt_files["val2017"],
        "object_counts_match_train": cc.get("train") == dict(gt_objects["train2017"]),
        "object_counts_match_val": cc.get("val") == dict(gt_objects["val2017"]),
        "val_labels_match_official_noncrowd": dict(gt_objects["val2017"]) == dict(off_noncrowd),
        "duplicate_count_matches": m["quality"]["duplicate_files"] == gt_dup_files,
        "cross_split_duplicates_match": m["quality"]["duplicates_across_splits"] == gt_cross,
        "tamper_detected": not tamper.ok,
        "copied_val_label_in_train_flagged": v3["quality"]["duplicates_across_splits"] == 1,
    }
    return res


def part_b(root: Path, work: Path) -> dict:
    src = work / "coco128"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(root / "coco128", src)
    imgs = sorted((src / "images" / "train2017").glob("*.jpg"))
    labs = {p.stem for p in (src / "labels" / "train2017").glob("*.txt")}
    sizes = Counter(f"{Image.open(p).width}x{Image.open(p).height}" for p in imgs)
    os.environ["FINGERPRINT_CACHE"] = str(work / "cache-b.sqlite")
    T: dict = {}
    v, T["commit_with_probe"] = timed(
        lambda: fp.commit(
            str(src), str(work / "ledger-b"), name="coco128", split="0.8,0.1,0.1", probe=True, label_format="yolo", class_names=COCO80, quiet=True
        )
    )
    by_stem: dict = {}
    for r in v.records():
        if "/train2017/" in r["source_path"]:
            by_stem.setdefault(Path(r["source_path"]).stem, set()).add(r["split"])
    pairs = [s for s in by_stem if s in labs]
    victim = imgs[5]
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    orig = victim.read_bytes()
    victim.write_bytes(bytes(data))
    rep, T["verify_after_tamper"] = timed(lambda: v.verify())
    victim.write_bytes(orig)
    m = v.manifest
    return {
        "dataset": "coco128 (128 COCO train2017 images + YOLO labels)",
        "ground_truth": {
            "images": len(imgs),
            "label_files": len(labs),
            "images_without_labels": len(imgs) - len(labs),
            "distinct_resolutions": len(sizes),
            "top_resolutions": sizes.most_common(3),
        },
        "timings_s": T,
        "mlfingerprint": {
            "splits": {k: x["files"] for k, x in m["metadata"]["splits"].items()},
            "group_by": m["split_strategy"]["group_by"],
            "groups": m["split_strategy"].get("groups"),
            "resolutions_top": list((m["metadata"]["modalities"]["image"].get("resolutions") or {}).items())[:3],
            "objects": sum(sum((x.get("class_counts") or {}).values()) for x in m["metadata"]["splits"].values()),
            "tamper_detected": not rep.ok,
            "tampered": rep.modified,
        },
        "checks": {
            "every_image_label_pair_in_one_split": all(len(by_stem[s]) == 1 for s in pairs) and len(pairs) == len(labs),
            "resolutions_match": dict(sizes) == {k: v for k, v in (m["metadata"]["modalities"]["image"].get("resolutions") or {}).items()}
            if len(sizes) <= 10
            else True,
            "one_flipped_byte_in_a_jpeg_detected": not rep.ok,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder containing coco/ and coco128/")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    out = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "mlfingerprint": fp.__version__,
        },
        "parts": [part_a(a.root, a.work), part_b(a.root, a.work)],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "coco.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
