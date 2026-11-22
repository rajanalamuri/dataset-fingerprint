"""Split detection (train/val/test from directory or file names) and stable,
group-aware hash splitting for datasets that don't have splits yet."""

from __future__ import annotations

import hashlib
import posixpath
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train", "training", "trn"),
    "val": ("val", "valid", "validation", "dev", "development"),
    "test": ("test", "testing", "tst", "eval", "evaluation", "holdout", "heldout"),
}
_ALIAS = {a: canon for canon, aliases in SPLIT_ALIASES.items() for a in aliases}
_TOKEN = re.compile(r"[-_.\s]")

DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
GroupBy = Union[None, str, Callable[[str], str]]


def canonical_split(name: str) -> Optional[str]:
    return _ALIAS.get(name.lower())


def detect_split(rel: str) -> Tuple[Optional[str], str]:
    """Return (split, path_without_split_dir).

    - directory form, at any depth:  train/cat/1.jpg, audio/val/x.wav
    - file-name form (HF / Kaggle):  train.csv, test-00001-of-00004.parquet
    """
    parts = rel.split("/")
    for i, p in enumerate(parts[:-1]):
        s = canonical_split(p) or canonical_split(re.sub(r"[-_]?\d+$", "", p))  # train2017, val_01
        if s:
            return s, "/".join(parts[:i] + parts[i + 1 :])
    stem = parts[-1].split(".", 1)[0]
    tokens = [t for t in _TOKEN.split(stem) if t]
    cands = [stem] + tokens[:1] + [tokens[-1].rstrip("0123456789")] if tokens else [stem]
    for c in cands:
        s = canonical_split(c)
        if s:
            return s, rel
    return None, rel


def parse_ratios(spec: Union[None, str, Dict[str, float], Sequence[float]]) -> Optional[Dict[str, float]]:
    """'0.8,0.1,0.1' | 'train=0.7,val=0.2,test=0.1' | [0.8, 0.2] | {'train': .9, 'test': .1}"""
    if spec is None:
        return None
    ratios: Dict[str, float] = {}
    if isinstance(spec, dict):
        ratios = {k: float(v) for k, v in spec.items()}
    else:
        items: List[Any] = [s.strip() for s in spec.split(",")] if isinstance(spec, str) else list(spec)
        default_names = ["train", "val", "test"] if len(items) == 3 else ["train", "test"] if len(items) == 2 else []
        for i, item in enumerate(items):
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                ratios[k.strip()] = float(v)
            else:
                if i >= len(default_names):
                    raise ValueError("Unnamed ratios must have 2 or 3 values; name them like train=0.7,val=0.3")
                ratios[default_names[i]] = float(item)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must be positive")
    return {k: v / total for k, v in ratios.items()}


def group_key(rel: str, group_by: GroupBy) -> str:
    """Files sharing a group key always land in the same split (no leakage of a
    speaker / video / patient / session across train and test)."""
    if group_by is None:
        return rel
    if callable(group_by):
        return str(group_by(rel))
    if group_by == "parent":
        return posixpath.dirname(rel) or rel
    if group_by == "stem":  # e.g. frame_001.jpg + frame_001.json + frame_001.pcd
        base = posixpath.basename(rel).split(".", 1)[0]
        return posixpath.join(posixpath.dirname(rel), base)
    m = re.search(group_by, rel)
    if not m:
        return rel
    return m.group(1) if m.groups() else m.group(0)


def hash_split(key: str, ratios: Dict[str, float], seed: str = "mlfingerprint") -> str:
    """Deterministic: the same key always maps to the same split, so adding new
    files in later versions never reshuffles existing ones."""
    h = int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:15], 16) / float(16**15)
    cum = 0.0
    names = list(ratios)
    for name in names:
        cum += ratios[name]
        if h < cum:
            return name
    return names[-1]


# --------------------------------------------------------------------------- #
#  Leakage-safe automatic grouping for sensor data
# --------------------------------------------------------------------------- #
_NUM_TAIL = re.compile(r"^(.*?)(\d+)$")
FRAME_BLOCK = 100  # consecutive frames of one stream that always stay together


def _stem(name: str) -> str:
    return name.split(".", 1)[0]


def _sequence_stats(paths: List[str], min_frames: int = 3) -> Dict[str, Tuple[int, int]]:
    """{dir: (n_frames, step)} for directories whose numbered files look like a *sequence*:
    densely numbered (frame_000001, 000002, ...) or regularly spaced (epoch-ms timestamps at a
    fixed rate). Sparse, irregular IDs - COCO's 000000000009.jpg, 000000000025.jpg, ImageNet
    ids - are independent samples, not frames, and are left alone."""
    nums: Dict[str, List[int]] = {}
    for p in paths:
        d, name = posixpath.split(p)
        m = _NUM_TAIL.match(_stem(name))
        if m:
            nums.setdefault(d, []).append(int(m.group(2)))
    out: Dict[str, Tuple[int, int]] = {}
    for d, xs in nums.items():
        u = sorted(set(xs))
        if len(xs) < min_frames or len(u) < 2:
            continue
        gaps = [b - a for a, b in zip(u, u[1:])]
        mean = sum(gaps) / len(gaps)
        cv = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean if mean else 0.0
        dense = (u[-1] - u[0] + 1) / len(u) <= 1.5
        if dense or cv < 0.5:
            out[d] = (len(xs), 1 if dense else max(1, sorted(gaps)[len(gaps) // 2]))
    return out


def sequence_dirs(paths: List[str], min_frames: int = 3) -> Dict[str, int]:
    """Directories that look like frame sequences. Returns {dir: n_frames}."""
    return {d: n for d, (n, _) in _sequence_stats(paths, min_frames).items()}


def sequence_steps(paths: List[str], min_frames: int = 3) -> Dict[str, int]:
    """Numbering step per sequence directory (1 for frame counters, e.g. 100 for 10 Hz epoch-ms)."""
    return {d: step for d, (_, step) in _sequence_stats(paths, min_frames).items()}


def _block(stem: str, size: int = FRAME_BLOCK, step: int = 1) -> str:
    m = _NUM_TAIL.match(stem)
    return f"{m.group(1)}#{int(m.group(2)) // (size * max(1, step))}" if m else stem


def auto_group_key(
    rest: str,
    seq_dirs: Dict[str, int],
    labeled: bool,
    level: int = 0,
    block: int = FRAME_BLOCK,
    steps: Optional[Dict[str, int]] = None,
) -> str:
    """Group so that nothing correlated is split apart:

    level 0  session/recording: everything before the first sensor folder
             drives/2026-05-08_run3/{camera,radar,imu}/...   -> 'drives/2026-05-08_run3'
    level 1  sequence: the folders after the sensor folder, shared across sensors
             {camera,radar}/seq_014/000123.*                  -> 'seq_014'
             or a frame-sequence directory                    -> its path
             or blocks of consecutive frames (100, shrinking if too few groups), synced by frame number across sensors
    level 2  single file
    """
    from .modality import sensor_dir_index

    if level >= 2:
        return "file:" + rest
    parts = rest.split("/")
    dirs, stem = parts[:-1], _stem(parts[-1])
    i = sensor_dir_index(rest)
    if i is not None:
        prefix, after = "/".join(dirs[:i]), "/".join(dirs[i + 1 :])
        if level == 0 and prefix:
            return "rec:" + prefix
        scope = "/".join(x for x in (prefix, after) if x)
        if after:
            return "seq:" + scope
        return "blk:" + (scope + "/" if scope else "") + _block(stem, block, (steps or {}).get("/".join(dirs), 1))
    parent = "/".join(dirs)
    if not labeled and parent in seq_dirs:
        if level == 0:
            return "seq:" + parent
        return "blk:" + parent + "/" + _block(stem, block, (steps or {}).get(parent, 1))
    return "file:" + (parent + "/" if parent else "") + stem


# --------------------------------------------------------------------------- #
#  Augmentation awareness
# --------------------------------------------------------------------------- #
# image <-> annotation pairing folders (YOLO, VOC, Cityscapes, masks): images/x.jpg + labels/x.txt
PAIR_DIRS = {
    "images",
    "image",
    "imgs",
    "img",
    "labels",
    "label",
    "masks",
    "mask",
    "annotations",
    "annotation",
    "anns",
    "ann",
    "gt",
    "gtfine",
    "groundtruth",
    "ground_truth",
    "jpegimages",
    "segmentationclass",
    "segmentationobject",
    "instances",
    "seg",
    "segmentation",
    "boxes",
    "bboxes",
    "keypoints",
    "depth_gt",
}
AUG_DIRS = {"aug", "augs", "augmented", "augmentation", "augmentations", "synthetic", "synth", "generated", "gen"}
AUG_TOKENS = {
    # vision
    "aug",
    "augmented",
    "flip",
    "hflip",
    "vflip",
    "mirror",
    "mirrored",
    "rot",
    "rotate",
    "rotated",
    "crop",
    "rcrop",
    "cropped",
    "blur",
    "blurred",
    "noise",
    "noisy",
    "gauss",
    "jitter",
    "colorjitter",
    "cj",
    "bright",
    "brightness",
    "contrast",
    "scale",
    "scaled",
    "shift",
    "shifted",
    "affine",
    "perspective",
    "cutout",
    "cutmix",
    "mixup",
    "mosaic",
    "erase",
    "grayscale",
    "gray",
    "hsv",
    "distort",
    "distorted",
    "elastic",
    "fog",
    "rain",
    "snow",
    "night",
    # audio
    "pitch",
    "pitchshift",
    "speed",
    "sp",
    "tempo",
    "stretch",
    "reverb",
    "rir",
    "specaug",
    "mix",
    "codec",
    "gain",
    "time_mask",
    "freq_mask",
    # text
    "para",
    "paraphrase",
    "bt",
    "backtrans",
    "backtranslated",
    "synonym",
    "eda",
    "tts",
    "synth",
    "synthetic",
    "gen",
    # sensors / generic
    "perturb",
    "perturbed",
    "dropout",
    "resample",
    "resampled",
    "warp",
    "warped",
    "drift",
    "copy",
    "dup",
}
_AUG_SUFFIX = re.compile(
    r"[_-](?:" + "|".join(sorted((re.escape(t) for t in AUG_TOKENS), key=len, reverse=True)) + r")(?:[_-]?[0-9][0-9p.]*)?$",
    re.IGNORECASE,
)


def strip_aug(stem: str) -> str:
    """'img_001_hflip_rot90' -> 'img_001', 'utt3_speed1' -> 'utt3', 'clip-aug2' -> 'clip'."""
    while True:
        m = _AUG_SUFFIX.search(stem)
        if not m or m.start() == 0:
            return stem
        stem = stem[: m.start()]


def aug_dir_index(rest: str) -> Optional[int]:
    for i, p in enumerate(rest.split("/")[:-1]):
        if p.lower() in AUG_DIRS:
            return i
    return None


def canonical_item(rest: str) -> Tuple[str, str, bool]:
    """(path_with_aug_dirs_removed_and_stem_stripped, base_stem, in_aug_dir)."""
    parts = rest.split("/")
    in_dir = any(p.lower() in AUG_DIRS for p in parts[:-1])
    dirs = [p for p in parts[:-1] if p.lower() not in AUG_DIRS and p.lower() not in PAIR_DIRS]
    name = parts[-1]
    stem, _, ext = name.partition(".")
    base = strip_aug(stem)
    return "/".join(dirs + [base + ("." + ext if ext else "")]), base, in_dir


# --------------------------------------------------------------------------- #
#  Time: for forecasting the split must follow time, never a hash
# --------------------------------------------------------------------------- #
_DATE = re.compile(r"(20\d{2}|19\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?(0[1-9]|[12]\d|3[01])(?:[T_ -]?([01]\d|2[0-3])[-_:]?([0-5]\d)(?:[-_:]?([0-5]\d))?)?")
_EPOCH = re.compile(r"(?<!\d)(1[5-9]\d{8}(?:\d{3}|\d{6}|\d{9})?)(?!\d)")


def time_from_path(rel: str) -> Optional[float]:
    """2026-05-08, 20260508_1430, .../2026/05/08/..., 1715000000[123] -> epoch seconds (UTC)."""
    from datetime import datetime, timezone

    m = _DATE.search(rel)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh, mm, ss = (int(m.group(i)) if m.group(i) else 0 for i in (4, 5, 6))
        try:
            return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    m = _EPOCH.search(rel)
    if m:
        v = m.group(1)
        return int(v) / {10: 1, 13: 1e3, 16: 1e6, 19: 1e9}[len(v)]
    return None


def parse_time(v) -> float:
    from datetime import datetime, timezone

    if isinstance(v, (int, float)):
        return float(v)
    dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def temporal_assign(
    group_times: Dict[str, float], group_sizes: Dict[str, int], ratios: Optional[Dict[str, float]] = None, cutoffs: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Oldest groups -> train, then val, newest -> test.
    cutoffs={'val': '2026-05-01', 'test': '2026-05-15'}: everything from that time on goes there."""
    order = sorted(group_times, key=lambda g: (group_times[g], g))
    out: Dict[str, str] = {}
    if cutoffs:
        bounds = sorted(((parse_time(t), name) for name, t in cutoffs.items()), key=lambda x: x[0])
        for g in order:
            name = "train"
            for t, n in bounds:
                if group_times[g] >= t:
                    name = n
            out[g] = name
        return out
    ratios = ratios or DEFAULT_RATIOS
    order_names = sorted(ratios, key=lambda n: {"train": 0, "val": 1, "test": 2}.get(n, 1))
    total = sum(group_sizes.values()) or 1
    edges, cum = [], 0.0
    for n in order_names:
        cum += ratios[n]
        edges.append((cum, n))
    seen = 0
    for g in order:
        frac = seen / total
        out[g] = next((n for e, n in edges if frac < e - 1e-12), order_names[-1])
        seen += group_sizes[g]
    return out
