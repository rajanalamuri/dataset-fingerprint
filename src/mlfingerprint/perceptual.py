"""Near-duplicate detection across splits with a 64-bit difference hash (dHash).

Exact-byte duplicates are caught by sha256. This catches the same picture after
re-encoding, resizing, light colour shifts or watermarking - the kind of copies that
put ~3% of CIFAR-10's test set into its training set (Barz & Denzler, ciFAIR).

Search is exact for the chosen threshold: split each hash into T+1 disjoint chunks;
two hashes within Hamming distance T must agree on at least one chunk (pigeonhole),
so only items sharing a chunk value are compared.

Requires Pillow (`pip install mlfingerprint[image]`); without it the check is skipped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import IO, Any, Dict, Iterable, List, Optional, Sequence, Tuple

BITS = 64
MAX_THRESHOLD = 10
HOT_BUCKET = 5000  # buckets bigger than this (e.g. thousands of flat images) are skipped and reported


def available() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def dhash(f: IO[bytes]) -> Optional[int]:
    """64-bit difference hash of an image file object, or None if it can't be decoded."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(f) as im:
            im.draft("L", (64, 64))  # fast JPEG decode at reduced size
            g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            px = list(g.tobytes())
    except Exception:
        return None
    h = 0
    for y in range(8):
        row = px[y * 9 : y * 9 + 9]
        for x in range(8):
            h = (h << 1) | (row[x + 1] > row[x])
    return h


def _chunks(threshold: int) -> List[Tuple[int, int]]:
    """T+1 disjoint (shift, mask) bit ranges covering all 64 bits."""
    m = threshold + 1
    out, start = [], 0
    for i in range(m):
        width = BITS // m + (1 if i < BITS % m else 0)
        out.append((start, (1 << width) - 1))
        start += width
    return out


def cross_split_pairs(items: Sequence[Tuple[str, int]], threshold: int = 4) -> Tuple[List[Tuple[int, int, int]], int]:
    """items: (split, hash) per file. Returns ([(i, j, distance)], skipped_hot_items) for every
    pair in *different* splits with Hamming distance <= threshold."""
    if not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(f"near-duplicate threshold must be 0..{MAX_THRESHOLD}")
    by_value: Dict[int, List[int]] = defaultdict(list)
    for i, (_, h) in enumerate(items):
        by_value[h].append(i)
    values = list(by_value)
    pairs: List[Tuple[int, int, int]] = []

    def emit(a: Sequence[int], b: Sequence[int], d: int) -> None:
        for i in a:
            for j in b:
                if i < j and items[i][0] != items[j][0]:
                    pairs.append((i, j, d))
                elif j < i and items[i][0] != items[j][0]:
                    pairs.append((j, i, d))

    for v in values:  # identical hashes (distance 0)
        idx = by_value[v]
        if len(idx) > 1 and len({items[i][0] for i in idx}) > 1:
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    if items[idx[a]][0] != items[idx[b]][0]:
                        pairs.append((idx[a], idx[b], 0))
    if threshold == 0:
        return pairs, 0

    splits_of = {v: {items[i][0] for i in by_value[v]} for v in values}
    seen: set = set()
    skipped = 0
    for shift, mask in _chunks(threshold):
        buckets: Dict[int, List[int]] = defaultdict(list)
        for vi, v in enumerate(values):
            buckets[(v >> shift) & mask].append(vi)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            if len(bucket) > HOT_BUCKET:
                skipped += len(bucket)
                continue
            for a in range(len(bucket)):
                va = values[bucket[a]]
                sa = splits_of[va]
                for b in range(a + 1, len(bucket)):
                    vb = values[bucket[b]]
                    if len(sa) == 1 and sa == splits_of[vb]:
                        continue
                    d = bin(va ^ vb).count("1")
                    if d > threshold:
                        continue
                    key = (va, vb) if va < vb else (vb, va)
                    if key in seen:
                        continue
                    seen.add(key)
                    emit(by_value[va], by_value[vb], d)
    return pairs, skipped


def summarize(paths: Sequence[str], splits: Sequence[str], pairs: Iterable[Tuple[int, int, int]], limit: int = 1000) -> Dict[str, Any]:
    """Per-split counts of files that have a near-duplicate in train, plus example pairs."""
    pairs = sorted(pairs, key=lambda p: (p[2], paths[p[0]], paths[p[1]]))
    in_train: Dict[str, set] = defaultdict(set)
    other: Dict[str, set] = defaultdict(set)
    for i, j, _ in pairs:
        si, sj = splits[i], splits[j]
        if sj == "train":
            in_train[si].add(i)
        elif si == "train":
            in_train[sj].add(j)
        else:
            (a, sa), (b, sb) = sorted(((i, si), (j, sj)), key=lambda t: t[1])
            other[f"{sa}~{sb}"].add(b)
    return {
        "pairs": len(pairs),
        "with_train_match": {k: len(v) for k, v in sorted(in_train.items())},
        "between_eval_splits": {k: len(v) for k, v in sorted(other.items())},
        "examples": [[paths[i], paths[j], d] for i, j, d in pairs[:limit]],
    }
