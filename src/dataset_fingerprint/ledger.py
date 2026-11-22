"""The Ledger: a folder (local, S3, Azure, GCS, Databricks, ...) of immutable,
manifest-described dataset versions.

    <root>/
      ledger.json
      <name>-v1.0.0/
        manifest.json        <- the contract between data pipeline and training
        index.jsonl.gz       <- one row per file: path, split, label, size, sha256, modality, meta
        splits/train.txt     <- absolute URIs, one per line, for any framework
        data/...             <- only in copy / hardlink mode
        usage.jsonl          <- models trained on this version
"""

from __future__ import annotations

import getpass
import gzip
import hashlib
import json
import os
import random
import re
import socket
import subprocess  # only `git` with fixed argv for provenance  # nosec B404
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from . import __version__, perceptual
from .diff import Diff, bump_semver, compute_diff, parse_semver
from .hashing import ALGORITHM, HashCache, digest, hash_stream
from .modality import detect_modality, sensor_keyword
from .probe import count_record_labels, count_yolo_labels, probe_file
from .splits import (
    DEFAULT_SEED,
    DEFAULT_RATIOS,
    GroupBy,
    auto_group_key,
    canonical_item,
    detect_split,
    group_key,
    hash_split,
    parse_ratios,
    parse_time,
    sequence_dirs,
    sequence_steps,
    temporal_assign,
    time_from_path,
)
from .splits import PAIR_DIRS
from .storage import Location, UnsafePathError, copy_file

SCHEMA = "dataset-fingerprint/manifest@1"
MANIFEST = "manifest.json"
INDEX = "index.jsonl.gz"
USAGE = "usage.jsonl"
LEDGER_META = "ledger.json"
TAGS = "tags.json"
MODES = ("reference", "copy", "hardlink", "cas")
Parent = Union["DatasetVersion", Dict[str, Any], str]
_NON_LABEL_DIRS = {
    "images",
    "image",
    "imgs",
    "img",
    "audio",
    "audios",
    "wav",
    "wavs",
    "video",
    "videos",
    "clips",
    "frames",
    "data",
    "raw",
    "text",
    "texts",
    "docs",
    "files",
    "samples",
    "labels",
    "annotations",
}

LabelFrom = Union[str, Callable[[str], Optional[str]], None]


class FingerprintError(RuntimeError):
    pass


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


class _Progress:
    def __init__(self, total: int, label: str, quiet: bool):
        self.total, self.label, self.n = total, label, 0
        self.on = not quiet and sys.stderr.isatty() and total > 0

    def tick(self) -> None:
        self.n += 1
        if self.on and (self.n == self.total or self.n % 50 == 0):
            print(f"\r  {self.label}: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)
            if self.n == self.total:
                print(file=sys.stderr)


def _who() -> Dict[str, Any]:
    info: Dict[str, Any] = {"tool": f"dataset-fingerprint {__version__}"}
    try:
        info["user"] = getpass.getuser()
    except Exception:  # no login name in some containers  # nosec B110
        pass
    info["host"] = socket.gethostname()
    for key, cmd in (("git_commit", ["git", "rev-parse", "HEAD"]), ("git_user", ["git", "config", "user.email"])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)  # fixed argv  # nosec B603
            if out.returncode == 0 and out.stdout.strip():
                info[key] = out.stdout.strip()
        except Exception:  # git is optional provenance  # nosec B110
            pass
    for env in ("CI_PIPELINE_URL", "GITHUB_RUN_ID", "DATABRICKS_RUNTIME_VERSION", "BUILD_BUILDID"):
        if os.environ.get(env):
            info.setdefault("ci", {})[env] = os.environ[env]
    return info


def _write_index(loc: Location, rel: str, rows: List[Dict[str, Any]]) -> None:
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    loc.write_bytes(rel, gzip.compress(body.encode("utf-8"), mtime=0))


def _read_index(loc: Location, rel: str) -> List[Dict[str, Any]]:
    with loc.open(rel, "rb") as f:
        raw = gzip.decompress(f.read()).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line]


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].split(".", 1)[0]


def _dir(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _mark_augmented(rows: List[Dict[str, Any]], known_originals: Optional[set] = None) -> None:
    """Flag augmented copies: files in aug/ synthetic/ folders, or files whose name is an
    existing file's name + an augmentation suffix (img_001_hflip.png next to img_001.png).
    Sets r['key_rest'] - the path of the original - used for grouping and labels."""
    info = []
    originals: set = set(known_originals or ())
    for r in rows:
        key_rest, base, in_dir = canonical_item(r["rest"])
        info.append((key_rest, base, in_dir))
        if not in_dir:
            originals.add(_stem(r["rest"]))
    for r, (key_rest, base, in_dir) in zip(rows, info):
        own = _stem(r["rest"])
        if in_dir or (base != own and base in originals):
            r["augmented"] = True
            r["aug_of"] = base
            r["key_rest"] = key_rest
        else:
            parts = r["rest"].split("/")
            r["key_rest"] = "/".join([p for p in parts[:-1] if p.lower() not in PAIR_DIRS] + parts[-1:])


def _norm_line_hashes(f, skip_header: bool) -> set:
    out = set()
    for i, raw in enumerate(f):
        if skip_header and i == 0:
            continue
        line = b" ".join(raw.strip().lower().split())
        if line:
            out.add(hashlib.blake2b(line, digest_size=8).digest())
    return out


def _leakage_checks(rows: List[Dict[str, Any]], parents: List["DatasetVersion"]) -> Dict[str, Any]:
    """Things that silently inflate evaluation scores."""
    warnings: List[str] = []
    real = [r for r in rows if not r.get("augmented") and r["split"] != "shared"]
    aug = [r for r in rows if r.get("augmented")]
    by_dir_stem: Dict[tuple, set] = defaultdict(set)
    stem_dirs: Dict[str, set] = defaultdict(set)
    stem_splits: Dict[str, set] = defaultdict(set)
    for r in real:
        st, d = _stem(r["key_rest"]), _dir(r["key_rest"])
        by_dir_stem[(d, st)].add(r["split"])
        stem_dirs[st].add(d)
        stem_splits[st].add(r["split"])

    def original_splits(r):
        base = r["aug_of"]
        hit = by_dir_stem.get((_dir(r["key_rest"]), base))
        if hit:
            return hit
        return stem_splits[base] if len(stem_dirs.get(base, ())) == 1 else None

    real_at: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in real:
        real_at[(_dir(r["key_rest"]), _stem(r["key_rest"]))].append(r)

    cross, eval_aug = [], []
    aug_leaked: set = set()  # evaluation files that share content with train through augmentation
    for r in aug:
        o = original_splits(r)
        if o and r["split"] not in o:
            cross.append(f"{r['source_path']} ({r['split']}) is an augmentation of a {'/'.join(sorted(o))} file")
            if r["split"] not in ("train", "shared") and "train" in o:
                aug_leaked.add(r["source_path"])
            elif r["split"] == "train":
                origs = real_at.get((_dir(r["key_rest"]), r["aug_of"])) or (
                    [x for x in real if _stem(x["key_rest"]) == r["aug_of"]] if len(stem_dirs.get(r["aug_of"], ())) == 1 else []
                )
                aug_leaked.update(x["source_path"] for x in origs if x["split"] not in ("train", "shared"))
        if r["split"] in ("val", "test"):
            eval_aug.append(r["source_path"])
    if cross:
        warnings.append(f"{len(cross)} augmented file(s) sit in a different split than their original - leakage, e.g. {cross[0]}")
    if eval_aug:
        warnings.append(f"{len(eval_aug)} augmented file(s) in val/test - evaluate on real data, e.g. {eval_aug[0]}")

    # exact duplicates
    sha_splits: Dict[str, set] = defaultdict(set)
    sha_paths: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if r["split"] != "shared":
            sha_splits[r["sha256"]].add(r["split"])
            sha_paths[r["sha256"]].append(r["source_path"])
    dup_cross = [h for h, sp in sha_splits.items() if len(sp) > 1]
    if dup_cross:
        ex = sha_paths[dup_cross[0]][:2]
        warnings.append(f"{len(dup_cross)} identical file(s) appear in more than one split, e.g. {ex}")

    # derived dataset vs its parents
    parent_leaks: List[str] = []
    for pv in parents:
        p_eval_sha = {r["sha256"] for r in pv.records() if r["split"] in ("val", "test")}
        p_norm: Dict[str, set] = defaultdict(set)
        p_stems: Dict[str, set] = defaultdict(set)
        p_dirs: Dict[str, set] = defaultdict(set)
        for r in pv.records():
            rest = r.get("rest", r["path"])
            norm = canonical_item(rest)[0].rsplit(".", 1)[0]
            p_norm[norm].add(r["split"])
            p_stems[_stem(rest)].add(r["split"])
            p_dirs[_stem(rest)].add(_dir(rest))
        for r in rows:
            if r["split"] != "train":
                continue
            if r["sha256"] in p_eval_sha:
                parent_leaks.append(f"{r['source_path']} is a copy of a {pv.version} val/test file")
            elif r.get("augmented"):
                own = (_dir(r["key_rest"]) + "/" if _dir(r["key_rest"]) else "") + r["aug_of"]
                sp = p_norm.get(own) or (p_stems.get(r["aug_of"]) if len(p_dirs.get(r["aug_of"], ())) == 1 else None)
                if sp and "train" not in sp and sp & {"val", "test"}:  # ambiguous names -> no false alarm
                    parent_leaks.append(f"{r['source_path']} augments a {pv.version} val/test file")
    if parent_leaks:
        warnings.append(f"{len(parent_leaks)} training file(s) come from a parent's val/test split, e.g. {parent_leaks[0]}")

    quality: Dict[str, Any] = {
        "duplicate_files": len(rows) - len({r["sha256"] for r in rows}),
        "duplicates_across_splits": len(dup_cross),
        "augmented_files": len(aug),
        "augmented_by_split": dict(Counter(r["split"] for r in aug)),
        "augmentation_cross_split": len(cross),
        "augmented_in_eval": len(eval_aug),
        "parent_eval_leaks": len(parent_leaks),
        "empty_files": sum(1 for r in rows if r["size"] == 0),
        "unreadable_files": sum(1 for r in rows if (r.get("meta") or {}).get("probe_error")),
    }
    examples = {
        "augmentation_cross_split": cross[:5],
        "augmented_in_eval": eval_aug[:5],
        "duplicates_across_splits": [sha_paths[h][:3] for h in dup_cross[:5]],
        "parent_eval_leaks": parent_leaks[:5],
    }
    quality["examples"] = {k: v for k, v in examples.items() if v}
    if quality["empty_files"]:
        warnings.append(f"{quality['empty_files']} empty file(s)")
    if quality["unreadable_files"]:
        warnings.append(f"{quality['unreadable_files']} file(s) could not be read by the probe (corrupt?)")
    return {"warnings": warnings, "quality": quality, "augmentation_leaks": bool(cross or parent_leaks), "aug_leaked": aug_leaked}


CONFIRMED = ("exact_copy", "augmented_copy", "temporal_overlap")  # provable from bytes, names or time
SUSPECTED = ("near_duplicate",)  # a perceptual-hash match: likely, not proven
NEAR_DUP_UNINFORMATIVE = 0.10  # flagging more than this share of a split means the hash can't tell samples apart


def _leak_metrics(rows: List[Dict[str, Any]], by_type: Dict[str, set], record_stats: Dict[str, tuple]) -> Dict[str, Any]:
    """leak_rate = share of val/test files that provably share content (byte copy, augmentation of the same
    file) or a time window with train. suspected_rate adds perceptual near-duplicates. Each file counts once."""
    eval_files: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if r["split"] not in ("train", "shared", "all") and not r.get("_annotation"):
            eval_files[r["split"]].append(r["source_path"])
    has_train = any(r["split"] == "train" and not r.get("_annotation") for r in rows)
    out: Dict[str, Any] = {
        "definition": "leak_rate: val/test files that are byte copies or augmentations of train files, or overlap "
        "train in time / val+test files; suspected_rate also counts perceptual near-duplicates"
    }
    if not has_train or not eval_files:
        out["leak_rate"] = None
        out["note"] = "needs a train split and at least one evaluation split"
        return out
    total = leaked_total = suspected_total = 0
    per: Dict[str, Any] = {}
    uninformative = []
    for sp, paths in sorted(eval_files.items()):
        pset = set(paths)
        hit = {t: by_type.get(t, set()) & pset for t in CONFIRMED + SUSPECTED}
        leaked = set().union(*(hit[t] for t in CONFIRMED))
        suspected = leaked.union(*(hit[t] for t in SUSPECTED))
        per[sp] = {
            "files": len(pset),
            "leaked": len(leaked),
            "leak_rate": round(len(leaked) / len(pset), 6),
            "effective_size": len(pset) - len(leaked),
            "by_type": {k: len(v) for k, v in hit.items() if v},
        }
        if "near_duplicate" in by_type:
            per[sp]["suspected"] = len(suspected)
            per[sp]["suspected_rate"] = round(len(suspected) / len(pset), 6)
            if len(hit["near_duplicate"]) > NEAR_DUP_UNINFORMATIVE * len(pset):
                uninformative.append(sp)
        if sp in record_stats:
            n, ov = record_stats[sp]
            per[sp]["records"] = n
            per[sp]["records_also_in_train"] = ov
            per[sp]["record_leak_rate"] = round(ov / n, 6) if n else 0.0
        total += len(pset)
        leaked_total += len(leaked)
        suspected_total += len(suspected)
    out["leak_rate"] = round(leaked_total / total, 6)
    out["leaked_files"] = leaked_total
    out["evaluation_files"] = total
    if "near_duplicate" in by_type:
        out["suspected_rate"] = round(suspected_total / total, 6)
        out["suspected_files"] = suspected_total
        if uninformative:
            out["near_duplicates_uninformative"] = uninformative
    out["by_split"] = per
    out["checked"] = sorted(t for t in CONFIRMED + SUSPECTED if t in by_type)
    return out


def _ledger_root_of(manifest_uri: str) -> str:
    if "://" in manifest_uri:
        return manifest_uri.rsplit("/", 2)[0]
    return os.path.dirname(os.path.dirname(manifest_uri))


def load_lock(lock: Dict[str, Any], storage_options: Optional[Dict[str, Any]] = None) -> "DatasetVersion":
    """Re-open the exact dataset version a model was trained on, from its stored lock."""
    if not lock.get("manifest_uri"):
        raise FingerprintError("lock has no manifest_uri (external reference?)")
    lg = Ledger(_ledger_root_of(lock["manifest_uri"]), name=lock.get("dataset"), storage_options=storage_options)
    v = lg.get(lock["dataset_version"])
    if lock.get("fingerprint") and v.fingerprint != lock["fingerprint"]:
        raise FingerprintError(f"{v.version} fingerprint differs from the lock - the ledger was altered")
    return v


# --------------------------------------------------------------------------- #
#  A single immutable version
# --------------------------------------------------------------------------- #
class DatasetVersion:
    """A committed dataset version. Read-only, except for the usage log."""

    def __init__(self, ledger: "Ledger", folder: str, manifest: Dict[str, Any]):
        self.ledger = ledger
        self.folder = folder
        self.manifest = manifest
        self._index: Optional[List[Dict[str, Any]]] = None

    # -- identity ------------------------------------------------------------
    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def semver(self) -> str:
        return self.manifest["semver"]

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]

    @property
    def uri(self) -> str:
        return self.ledger.loc.uri(self.folder)

    @property
    def data_location(self) -> Location:
        return Location(self.manifest["data_location"], self.ledger.storage_options)

    @property
    def splits(self) -> List[str]:
        return list(self.manifest["metadata"]["splits"])

    def __getitem__(self, key: str) -> Any:
        return self.manifest[key]

    def __repr__(self) -> str:
        m = self.manifest["metadata"]
        return f"<DatasetVersion {self.version} files={m['total_files']} splits={self.splits} {self.fingerprint[:19]}>"

    # -- data access ------------------------------------------------------------
    def records(self, split: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._index is None:
            self._index = _read_index(self.ledger.loc, f"{self.folder}/{INDEX}")
        return [r for r in self._index if split is None or r["split"] == split]

    def files(self, split: Optional[str] = None) -> List[str]:
        """Absolute URIs/paths of the files in a split (or all)."""
        loc = self.data_location
        return [loc.uri(r["path"]) for r in self.records(split)]

    def samples(self, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Yield {'uri', 'label', 'modality', 'split', ...meta} - feed into any DataLoader."""
        loc = self.data_location
        for r in self.records(split):
            yield {"uri": loc.uri(r["path"]), "label": r.get("label"), "modality": r["modality"], "split": r["split"], **(r.get("meta") or {})}

    def open(self, record_or_path: Union[Dict[str, Any], str], mode: str = "rb"):
        path = record_or_path["path"] if isinstance(record_or_path, dict) else record_or_path
        return self.data_location.open(path, mode)

    # -- training traceability --------------------------------------------------
    def lock(self) -> Dict[str, Any]:
        """Small dict to store with a model / experiment run."""
        return {
            "dataset": self.manifest["name"],
            "dataset_version": self.version,
            "fingerprint": self.fingerprint,
            "manifest_uri": self.ledger.loc.uri(f"{self.folder}/{MANIFEST}"),
            "split_checksums": {k: v for k, v in self.manifest["checksums"].items() if k.endswith("_split")},
        }

    def split_for_time(self, t: Any) -> str:
        """For row-level time splits recorded with row_split=: which split does timestamp t belong to?"""
        rs = self.manifest.get("row_split")
        if not rs:
            raise FingerprintError(f"{self.version} has no row_split")
        x = parse_time(t)
        if rs.get("test_from") and x >= parse_time(rs["test_from"]):
            return "test"
        if rs.get("val_from") and x >= parse_time(rs["val_from"]):
            return "val"
        return "train"

    def tags_dict(self) -> Dict[str, str]:
        """Flat key/values for MLflow tags, feature-store table properties, W&B config, ..."""
        return {"dataset": self.manifest["name"], "dataset_version": self.version, "dataset_fingerprint": self.fingerprint}

    def record_model(
        self,
        model_version: str,
        metrics: Optional[Dict[str, Any]] = None,
        recipe: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Append "model X was trained on this version" to usage.jsonl.

        recipe: on-the-fly augmentation / preprocessing config (it isn't in the data, so
        record it here); seed: RNG seed. Together with the fingerprint the run is reproducible."""
        entry = {
            "model_version": model_version,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            **extra,
            "dataset": self.lock(),
        }
        if recipe is not None:
            entry["recipe"] = recipe
        if seed is not None:
            entry["seed"] = seed
        rel = f"{self.folder}/{USAGE}"
        loc = self.ledger.loc
        existing = loc.read_text(rel) if loc.exists(rel) else ""
        loc.write_text(rel, existing + json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def usage(self) -> List[Dict[str, Any]]:
        rel = f"{self.folder}/{USAGE}"
        if not self.ledger.loc.exists(rel):
            return []
        return [json.loads(x) for x in self.ledger.loc.read_text(rel).splitlines() if x.strip()]

    # -- lineage -----------------------------------------------------------------
    def parents(self) -> List[Dict[str, Any]]:
        return list((self.manifest.get("lineage") or {}).get("parents") or [])

    def trace(self, depth: int = 10) -> Dict[str, Any]:
        """Full ancestry tree: this version -> parents -> their parents (external refs are leaves)."""
        node: Dict[str, Any] = {
            "dataset_version": self.version,
            "fingerprint": self.fingerprint,
            "message": self.manifest.get("message"),
            "parents": [],
        }
        if depth <= 0:
            return node
        for p in self.parents():
            if p.get("manifest_uri"):
                try:
                    node["parents"].append(load_lock(p, self.ledger.storage_options).trace(depth - 1))
                except Exception as e:
                    node["parents"].append({**p, "error": str(e)})
            else:
                node["parents"].append(p)
        return node

    def check_lineage(self) -> List[str]:
        """Warn if a parent is unreachable, was altered, or has newer versions since we derived."""
        out = []
        for p in self.parents():
            if not p.get("manifest_uri"):
                continue
            try:
                pv = load_lock(p, self.ledger.storage_options)
            except Exception as e:
                out.append(f"parent {p.get('dataset_version')}: {e}")
                continue
            latest = pv.ledger.latest()
            if latest and latest.version != pv.version:
                out.append(f"stale: derived from {pv.version}, parent now at {latest.version}")
        return out

    # -- retrieval helpers ---------------------------------------------------------
    def checkout(
        self, dest: str, split: Optional[str] = None, mode: str = "copy", workers: int = 16, storage_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Materialise the version as <dest>/<split>/<path> (from any mode, incl. cas)."""
        if mode not in ("copy", "hardlink", "symlink"):
            raise FingerprintError("checkout mode must be copy | hardlink | symlink")
        tampered = self.index_tampered()
        if tampered:
            raise FingerprintError(f"refusing checkout: index does not match manifest checksums for {tampered}")
        src, dst = self.data_location, Location(dest, storage_options)
        rows = self.records(split)

        def one(r):
            rel = f"{r['split']}/{r.get('rest', r['source_path'])}"
            if mode == "symlink" and src.is_local and dst.is_local:
                target = dst.local_path(rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if not os.path.lexists(target):
                    os.symlink(src.local_path(r["path"]), target)
            else:
                copy_file(src, r["path"], dst, rel, "hardlink" if mode == "hardlink" else "copy")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(one, rows))
        dst.write_text(".dataset-fingerprint.json", json.dumps(self.lock(), indent=2))
        return dst.uri()

    def index_tampered(self) -> List[str]:
        """Splits whose index rows no longer match the manifest's split checksums (cheap: no file reads)."""
        by_split: Dict[str, List] = defaultdict(list)
        for r in self.records():
            by_split[r["split"]].append((r["path"], r["sha256"]))
        return [s for s, items in by_split.items() if self.manifest["checksums"].get(f"{s}_split") != digest(items)]

    def to_torch(self, split: str, loader: Optional[Callable[[Any, Dict[str, Any]], Any]] = None, transform: Optional[Callable[[Any], Any]] = None):
        """A torch Dataset over a split. loader(file_obj, record) -> sample (default: raw bytes)."""
        import torch.utils.data as tud  # lazy, optional dependency

        records, ds = self.records(split), self

        class _FingerprintDataset(tud.Dataset):
            fingerprint = ds.fingerprint

            def __len__(self):
                return len(records)

            def __getitem__(self, i):
                r = records[i]
                with ds.open(r) as f:
                    x = loader(f, r) if loader else f.read()
                return (transform(x) if transform else x), r.get("label")

        return _FingerprintDataset()

    def log_mlflow(self) -> None:
        """Log lock + tags to the active MLflow run (pip install mlflow)."""
        import mlflow  # optional dependency

        mlflow.set_tags(self.tags_dict())
        mlflow.log_dict(self.lock(), "dataset_lock.json")

    # -- integrity ----------------------------------------------------------------
    def verify(self, sample: Optional[float] = None, workers: int = 8, quiet: bool = True, quick: bool = False) -> "VerifyReport":
        return self.ledger.verify(self, sample=sample, workers=workers, quiet=quiet, quick=quick)

    def summary(self) -> str:
        m, md = self.manifest, self.manifest["metadata"]
        lines = [
            f"{m['version']}  ({m['created_at']})",
            f"  message     : {m.get('message') or '-'}",
            f"  fingerprint : {m['fingerprint']}",
            f"  data        : {m['data_location']}  [{m['source']['mode']}]",
            f"  files       : {md['total_files']:,}   size: {_human(md['total_bytes'])}",
            "  splits      : " + "  ".join(f"{s}={v['files']:,} ({v['fraction']:.0%})" for s, v in md["splits"].items()),
            "  modalities  : "
            + "  ".join(
                f"{k}={v['files']:,}"
                + (f" ({_dur(v['duration_sec'])})" if v.get("duration_sec") else "")
                + (f" ({v['records']:,} rec)" if v.get("records") else "")
                for k, v in md["modalities"].items()
            ),
        ]
        if md.get("class_distribution"):
            top = list(md["class_distribution"].items())[:8]
            lines.append(
                "  classes     : "
                + "  ".join(f"{k}={v:.1%}" for k, v in top)
                + (f"  (+{md['num_classes'] - 8} more)" if md.get("num_classes", 0) > 8 else "")
            )
        lin = m.get("lineage") or {}
        if lin.get("previous_version"):
            ch = lin.get("changes", {})
            lines.append(
                f"  lineage     : from {lin['previous_version']} ({lin.get('bump')}): "
                f"+{ch.get('added', 0)} -{ch.get('removed', 0)} ~{ch.get('modified', 0)}"
            )
        for p in lin.get("parents") or []:
            lines.append("  parent      : " + (p.get("dataset_version") or json.dumps(p)))
        q = m.get("quality") or {}
        lk = q.get("leakage") or {}
        if lk.get("leak_rate") is not None:
            parts = [f"{sp} {v['leaked']:,}/{v['files']:,} (effective {v['effective_size']:,})" for sp, v in lk.get("by_split", {}).items()]
            lines.append(f"  leak rate   : {lk['leak_rate']:.2%}  " + "  ".join(parts))
            if lk.get("suspected_rate") is not None:
                lines.append(f"  suspected   : {lk['suspected_rate']:.2%} incl. near-duplicates")
        if q.get("augmented_files"):
            lines.append(
                f"  augmented   : {q['augmented_files']:,} files  " + "  ".join(f"{k}={v}" for k, v in q.get("augmented_by_split", {}).items())
            )
        for t in m.get("transformations") or []:
            lines.append(f"  transform   : {t if isinstance(t, str) else json.dumps(t)}")
        for w in lin.get("warnings") or []:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class VerifyReport:
    version: str
    checked: int = 0
    ok_files: int = 0
    missing: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    index_tampered: List[str] = field(default_factory=list)  # splits whose digest != manifest
    unsafe: List[str] = field(default_factory=list)  # index paths that try to escape the data root

    @property
    def ok(self) -> bool:
        return not (self.missing or self.modified or self.index_tampered or self.unsafe)

    def render(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [f"verify {self.version}: {status}  ({self.ok_files}/{self.checked} files intact)"]
        for p in self.missing[:20]:
            lines.append(f"  missing  : {p}")
        for p in self.modified[:20]:
            lines.append(f"  modified : {p}")
        for s in self.index_tampered:
            lines.append(f"  index/manifest mismatch for split '{s}'")
        for p in self.unsafe[:20]:
            lines.append(f"  UNSAFE path in index (not read): {p!r}")
        return "\n".join(lines)


def _dur(sec: float) -> str:
    if sec < 120:
        return f"{sec:.1f}s"
    if sec < 7200:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.1f}h"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} PB"


# --------------------------------------------------------------------------- #
#  The ledger
# --------------------------------------------------------------------------- #
class Ledger:
    def __init__(self, root: Optional[str] = None, name: Optional[str] = None, storage_options: Optional[Dict[str, Any]] = None):
        root = root or os.environ.get("FINGERPRINT_ROOT")
        if not root:
            raise FingerprintError("No ledger root given (pass root=... or set FINGERPRINT_ROOT).")
        self.storage_options = storage_options or {}
        self.loc = Location(root, self.storage_options)
        meta = json.loads(self.loc.read_text(LEDGER_META)) if self.loc.exists(LEDGER_META) else {}
        self.name = name or meta.get("name") or "dataset"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise FingerprintError(f"invalid dataset name {self.name!r}")

    def __repr__(self) -> str:
        return f"<Ledger {self.name} @ {self.loc.uri()} versions={len(self.versions())}>"

    # -- discovery -------------------------------------------------------------
    def _folder(self, semver: str) -> str:
        return f"{self.name}-v{semver}"

    def versions(self) -> List[str]:
        """All committed versions, oldest first (e.g. ['dataset-v1.0.0', ...])."""
        pat = re.compile(rf"^{re.escape(self.name)}-v(\d+\.\d+\.\d+)$")
        found = []
        for entry in self.loc.listdir():
            m = pat.match(entry)
            if m and self.loc.exists(f"{entry}/{MANIFEST}"):
                found.append((parse_semver(m.group(1)), entry))
        return [e for _, e in sorted(found)]

    def latest(self) -> Optional["DatasetVersion"]:
        v = self.versions()
        return self.get(v[-1]) if v else None

    def _resolve(self, version: Union[str, DatasetVersion, None]) -> str:
        if isinstance(version, DatasetVersion):
            return version.folder
        vs = self.versions()
        if not vs:
            raise FingerprintError(f"ledger {self.loc.uri()} has no versions yet")
        if version in (None, "latest"):
            return vs[-1]
        tags = self.tags()
        if isinstance(version, str) and version in tags:
            return self._resolve(tags[version]["version"])
        if isinstance(version, str) and version.startswith("latest~"):
            return vs[-1 - int(version.split("~", 1)[1])]
        try:
            sv = "%d.%d.%d" % parse_semver(str(version))
        except ValueError:
            raise FingerprintError(f"unknown version {version!r}") from None
        folder = self._folder(sv)
        if folder not in vs:
            raise FingerprintError(f"version {sv} not found. Available: {', '.join(vs)}")
        return folder

    def get(self, version: Union[str, DatasetVersion, None] = "latest") -> DatasetVersion:
        folder = self._resolve(version)
        return DatasetVersion(self, folder, json.loads(self.loc.read_text(f"{folder}/{MANIFEST}")))

    # -- tags: mutable, human names for immutable versions ('prod', 'golden', 'paper-v2')
    def tags(self) -> Dict[str, Dict[str, Any]]:
        return json.loads(self.loc.read_text(TAGS)) if self.loc.exists(TAGS) else {}

    def tag(self, version: Union[str, DatasetVersion], name: str, message: str = "") -> Dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name or ""):
            raise FingerprintError(f"invalid tag name {name!r}")
        if re.search(r"\d+\.\d+\.\d+$", name) or name.startswith("latest"):
            raise FingerprintError(f"tag {name!r} looks like a version")
        v = version if isinstance(version, DatasetVersion) else self.get(version)
        tags = self.tags()
        prev = tags.get(name)
        entry = {
            "version": v.version,
            "fingerprint": v.fingerprint,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "by": _who().get("user"),
            "history": ((prev or {}).get("history") or []) + ([{k: prev[k] for k in ("version", "updated_at")}] if prev else []),
        }
        tags[name] = entry
        self.loc.write_text(TAGS, json.dumps(tags, indent=2) + "\n")
        return entry

    def untag(self, name: str) -> None:
        tags = self.tags()
        tags.pop(name, None)
        self.loc.write_text(TAGS, json.dumps(tags, indent=2) + "\n")

    def _resolve_parents(self, parents):
        locks, versions = [], []
        for p in parents or []:
            if isinstance(p, DatasetVersion):
                locks.append(p.lock())
                versions.append(p)
            elif isinstance(p, str):
                root, sep, ver = p.rpartition("@")
                if not sep or "/" in ver:
                    root, ver = p, "latest"
                pv = Ledger(root, storage_options=self.storage_options).get(ver)
                locks.append(pv.lock())
                versions.append(pv)
            elif isinstance(p, dict) and p.get("manifest_uri"):
                pv = load_lock(p, self.storage_options)
                locks.append(pv.lock())
                versions.append(pv)
            elif isinstance(p, dict):
                locks.append({"type": "external", **p})
            else:
                raise FingerprintError(f"unsupported parent: {p!r}")
        return locks, versions

    def _claim(self, folder: str, commit_id: str) -> str:
        """Best-effort exclusive claim so two CI jobs can't write the same version."""
        rel = f".claims/{folder}"
        if self.loc.is_local:
            path = self.loc._p(rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = datetime.now().timestamp() - os.path.getmtime(path)
                if age < 3600:
                    raise FingerprintError(f"{folder} is being committed by another process (claim {path})") from None
                fd = os.open(path, os.O_WRONLY | os.O_TRUNC)  # stale claim from a crashed run
            with os.fdopen(fd, "w") as f:
                f.write(commit_id)
        else:
            # Object stores have no O_EXCL through fsspec: claim, then read back. Best effort -
            # run one committer per dataset in CI (see SECURITY.md / concurrency groups).
            conflict = FingerprintError(f"{folder} is being committed by another process")
            try:
                if self.loc.exists(rel) and self.loc.read_text(rel) != commit_id:
                    raise conflict
                self.loc.write_text(rel, commit_id)
                self.loc.refresh()
                if self.loc.read_text(rel) != commit_id:
                    raise conflict
            except FingerprintError:
                raise
            except Exception as e:  # claim vanished / overwritten mid-read
                raise conflict from e
        return rel

    def _release(self, rel: str) -> None:
        try:
            if self.loc.is_local:
                os.remove(self.loc._p(rel))
            else:
                self.loc.fs.rm(self.loc._p(rel))
        except Exception:  # read-only delete permissions etc.; a committed version never re-claims  # nosec B110
            pass

    def log(self) -> List[DatasetVersion]:
        return [self.get(v) for v in self.versions()]

    def diff(self, a: Union[str, DatasetVersion, None], b: Union[str, DatasetVersion, None] = "latest") -> Diff:
        va, vb = self.get(a), self.get(b)
        return compute_diff(va.version, va.records(), vb.version, vb.records())

    # -- commit ----------------------------------------------------------------
    def commit(
        self,
        source: str,
        *,
        message: str = "",
        version: Optional[str] = None,
        bump: str = "auto",
        transformations: Sequence[Union[str, Dict[str, Any]]] = (),
        source_batches: Optional[Sequence[str]] = None,
        mode: str = "reference",
        split: Union[str, Dict[str, float], Sequence[float], None] = "auto",
        group_by: GroupBy = "auto",
        seed: str = DEFAULT_SEED,
        modality_rules: Optional[Dict[str, Any]] = None,
        label_from: LabelFrom = "auto",
        label_column: Optional[str] = None,
        label_format: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        probe: bool = False,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        workers: int = 16,
        cache: bool = True,
        allow_empty: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        source_storage_options: Optional[Dict[str, Any]] = None,
        parents: Optional[Sequence[Parent]] = None,
        augmentation: str = "warn",
        time_cutoffs: Optional[Dict[str, Any]] = None,
        row_split: Optional[Dict[str, Any]] = None,
        content_checks: bool = True,
        near_duplicates: Union[bool, int] = False,
        max_leak_rate: Optional[float] = None,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> DatasetVersion:
        """Scan `source`, fingerprint every file, and write a new immutable version.

        mode     : 'reference' (manifest points at source, zero copy - verify() detects drift),
                   'copy' (snapshot into the version folder), 'hardlink' (local snapshot, no extra space),
                   'cas' (content-addressed snapshot in <ledger>/objects - identical files stored once
                   across all versions).
        split    : 'auto'  -> use train/val/test dirs or file names if present, else hash-split 80/10/10
                   'none'  -> keep everything in one 'all' split
                   '0.7,0.2,0.1' / {'train':.9,'test':.1} -> stable hash split (overrides detection)
        group_by : what must stay together in one split. 'auto' (default) keeps each recording /
                   session / frame-sequence together and time-syncs sensors; or 'parent', 'stem',
                   'file', a regex, or a callable(path) -> key.
        modality_rules: {"radar": ["*/mmwave/**", "*.rdc"]} - override modality detection.
        label_from: 'auto' (ImageFolder-style class dirs), 'parent', 'none', or callable(path)->label.
        label_column: count record-level labels inside csv/tsv/jsonl files (NLP / tabular).
        label_format: 'yolo' counts objects per class in YOLO/darknet .txt label files (detection,
                   segmentation); class_names maps class ids to names (e.g. the 80 COCO classes).
        parents  : datasets this one was derived from (augmented / filtered / relabelled sets):
                   DatasetVersion objects, lock dicts, "root@version" strings, or external refs
                   such as {"type": "delta", "table": "main.ml.frames", "version": 42}.
        augmentation: 'warn' | 'error' | 'ignore' - what to do when augmented copies leak across
                   splits or sit in evaluation splits.
        split='temporal' / 'temporal:0.7,0.15,0.15' / time_cutoffs={'val': '2026-05-01', 'test': '2026-05-15'}
                   forecasting-safe: oldest recordings -> train, newest -> test. Time comes from probed
                   timestamp columns (probe=True), dates/epochs in paths, or local mtimes.
        row_split: for ONE long series file: {'column': 'timestamp', 'val_from': ..., 'test_from': ...}
                   recorded in the manifest; use ds.split_for_time(t) in your loader.
        content_checks: hash every record of text/csv/jsonl files (<=256 MB each) to catch the same
                   sentence / row in train and test.
        near_duplicates: True (Hamming <= 4) or an int 0..10 - perceptual-hash every image and flag
                   re-encoded / resized / colour-shifted copies of the same picture in different splits
                   (needs Pillow). The 64-bit dHash is cached by sha256, so re-commits are cheap.
        max_leak_rate: refuse to commit (FingerprintError) if more than this fraction of val/test files
                   leak from train, e.g. 0.005 for 0.5%. The rate is always stored in manifest.quality.leakage.
        """
        if mode not in MODES:
            raise FingerprintError("mode must be " + " | ".join(MODES))
        if label_format not in (None, "yolo"):
            raise FingerprintError("label_format must be None or 'yolo'")
        if augmentation not in ("warn", "error", "ignore"):
            raise FingerprintError("augmentation must be warn | error | ignore")
        nd_threshold = None if near_duplicates is False or near_duplicates is None else (4 if near_duplicates is True else int(near_duplicates))
        if nd_threshold is not None and not 0 <= nd_threshold <= perceptual.MAX_THRESHOLD:
            raise FingerprintError(f"near_duplicates threshold must be 0..{perceptual.MAX_THRESHOLD}")
        if nd_threshold is not None and not perceptual.available():
            _log("dataset-fingerprint: WARNING near_duplicates needs Pillow (pip install dataset-fingerprint[image]) - skipped", quiet)
            nd_threshold = None
        parent_locks, parent_versions = self._resolve_parents(parents)
        src = Location(source, source_storage_options if source_storage_options is not None else self.storage_options)
        if not src.exists():
            raise FingerprintError(f"source not found: {source}")

        skip = [self.loc.root] if (self.loc.is_local and src.is_local) else []
        entries = list(src.walk(include=include, exclude=exclude, skip_dirs=skip))
        for sl in getattr(src, "skipped_symlinks", [])[:5]:
            _log(f"dataset-fingerprint: WARNING skipped symlink pointing outside the source: {sl}", quiet)
        if not entries:
            raise FingerprintError(f"no files found under {src.uri()}")
        _log(f"dataset-fingerprint: scanning {len(entries):,} files in {src.uri()}", quiet)

        # 1. splits ------------------------------------------------------------
        temporal = bool(time_cutoffs) or (isinstance(split, str) and split.startswith("temporal"))
        t_ratios = None
        if temporal and isinstance(split, str) and ":" in split:
            t_ratios = parse_ratios(split.split(":", 1)[1])
        forced = None if (temporal or split in ("auto", "none", None)) else parse_ratios(split)
        rows: List[Dict[str, Any]] = []
        for e in entries:
            detected, rest = detect_split(e.rel)
            rows.append(
                {
                    "source_path": e.rel,
                    "rest": rest,
                    "size": e.size,
                    "_mtime": e.mtime,
                    "_stamp": e.stamp,
                    "split": detected,
                    "modality": detect_modality(e.rel, modality_rules),
                }
            )
        has_dirs = any(r["split"] for r in rows)
        _mark_augmented(rows, {_stem(r.get("rest", r["path"])) for pv in parent_versions for r in pv.records()})
        # labels first: class folders must never be mistaken for recordings
        self._assign_labels(rows, label_from)
        if temporal and group_by == "auto":
            # time order itself prevents leakage; only keep files of one recording (same moment,
            # several sensors) together - never lump a whole series into one group
            from .splits import auto_group_key as _agk

            keys = [
                k if k.startswith("rec:") else "file:" + r["key_rest"] for r, k in zip(rows, (_agk(r["key_rest"], {}, "label" in r, 0) for r in rows))
            ]
            group_desc = "auto:recording-or-file"
        else:
            keys, group_desc = self._group_keys(rows, group_by, quiet)
        if temporal:
            strategy: Dict[str, Any] = {"type": "temporal", "group_by": group_desc}
            if time_cutoffs:
                strategy["cutoffs"] = {k: str(v) for k, v in time_cutoffs.items()}
            else:
                strategy["ratios"] = t_ratios or DEFAULT_RATIOS
            for r in rows:
                r["split"] = None  # assigned after probing, when timestamps are known
        elif forced:
            strategy = {"type": "hash", "ratios": forced, "group_by": group_desc, "seed": seed}
            for r, k in zip(rows, keys):
                r["split"] = hash_split(k, forced, seed)
        elif split == "none":
            strategy = {"type": "none"}
            for r in rows:
                r["split"] = "all"
        elif has_dirs:
            strategy = {"type": "detected"}
            unsplit = [r for r in rows if not r["split"]]
            if unsplit:  # e.g. top-level README, label maps -> shared metadata, not training data
                for r in unsplit:
                    r["split"] = "shared"
        else:
            strategy = {"type": "hash", "ratios": DEFAULT_RATIOS, "group_by": group_desc, "seed": seed}
            for r, k in zip(rows, keys):
                r["split"] = hash_split(k, DEFAULT_RATIOS, seed)
        if strategy.get("type") in ("hash", "temporal"):
            strategy["groups"] = len(set(keys))

        # 3. layout inside the version --------------------------------------------

        # 4. hash (+probe) in parallel ----------------------------------------------
        hc = HashCache() if cache else None
        prog = _Progress(len(rows), "hashing", quiet)

        def work(r: Dict[str, Any]) -> None:
            uri = src.uri(r["source_path"])
            sha = hc.get(uri, r["size"], r["_stamp"]) if hc else None
            if sha is None:
                with src.open(r["source_path"], "rb") as f:
                    sha = hash_stream(f)
                if hc:
                    hc.put(uri, r["size"], r["_stamp"], sha)
            r["sha256"] = sha
            if nd_threshold is not None and r["modality"] == "image" and r["split"] != "shared":
                ph = hc.get_ph(sha) if hc else None
                if ph is None:
                    with src.open(r["source_path"], "rb") as f:
                        ph = perceptual.dhash(f)
                    if ph is not None and hc:
                        hc.put_ph(sha, ph)
                if ph is not None:
                    r.setdefault("meta", {})["dhash"] = "%016x" % ph
            if probe:
                meta = probe_file(src, r["source_path"], r["modality"])
                if meta:
                    r["meta"] = meta
            ext = r["source_path"].rsplit(".", 1)[-1].lower()
            yolo = label_format == "yolo" and ext == "txt"
            if yolo:  # box coordinates are not "records": never line-hash them for contamination
                r["_annotation"] = True  # a companion of an image, not a sample: not counted in the leak rate
                yc = count_yolo_labels(src, r["source_path"])
                if yc:
                    names = list(class_names or [])
                    r.setdefault("meta", {})["label_counts"] = {(names[k] if k < len(names) else str(k)): v for k, v in sorted(yc.items())}
            elif content_checks and ext in ("jsonl", "ndjson", "csv", "tsv", "txt") and r["size"] <= (256 << 20):
                with src.open(r["source_path"], "rb") as f:
                    r["_lines"] = _norm_line_hashes(f, skip_header=ext in ("csv", "tsv"))
            if label_column and r["modality"] in ("text", "tabular"):
                c = count_record_labels(src, r["source_path"], label_column)
                if c:
                    r.setdefault("meta", {})["label_counts"] = dict(c)
            prog.tick()

        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                list(ex.map(work, rows))
        finally:
            if hc:
                hc.close()

        # 4a. temporal assignment (needs probed timestamps) ---------------------------
        if temporal:
            gt: Dict[str, float] = {}
            gs: Counter = Counter()
            missing = []
            for r, k in zip(rows, keys):
                t = (r.get("meta") or {}).get("start")
                if t is None:
                    t = time_from_path(r["source_path"])
                if t is None and src.is_local and r["_mtime"]:
                    t = r["_mtime"]
                if t is None:
                    missing.append(r["source_path"])
                    continue
                r["_t"] = t
                gt[k] = min(gt.get(k, t), t)
                gs[k] += 1
            if missing:
                raise FingerprintError(
                    f"temporal split needs a time for every file; none for e.g. {missing[0]} "
                    "(use probe=True for timestamp columns, or put dates in paths)"
                )
            assign = temporal_assign(gt, gs, strategy.get("ratios"), time_cutoffs)
            for r, k in zip(rows, keys):
                r["split"] = assign[k]
            spans: Dict[str, List[tuple]] = defaultdict(list)
            for r in rows:
                spans[r["split"]].append((r["_t"], r["_t"] + ((r.get("meta") or {}).get("duration_sec") or 0)))
            iso = lambda x: datetime.fromtimestamp(x, timezone.utc).isoformat(timespec="seconds")  # noqa: E731
            strategy["time_ranges"] = {sp: {"start": iso(min(a for a, _ in v)), "end": iso(max(b for _, b in v))} for sp, v in spans.items()}

        # 4b. layout inside the version ------------------------------------------------
        if mode in ("copy", "hardlink"):
            for r in rows:
                r["path"] = f"data/{r['split']}/{r['rest']}"
        elif mode == "reference":
            for r in rows:
                r["path"] = r["source_path"]
        if mode == "cas":
            for r in rows:
                ext = r["source_path"].rsplit(".", 1)[-1] if "." in r["source_path"].rsplit("/", 1)[-1] else ""
                r["path"] = f"objects/{r['sha256'][:2]}/{r['sha256']}" + (f".{ext.lower()}" if ext else "")

        # 4b. leakage + quality checks ---------------------------------------------------
        checks = _leakage_checks(rows, parent_versions)
        for w in checks["warnings"]:
            _log(f"dataset-fingerprint: WARNING {w}", quiet)
        if augmentation == "error" and checks["augmentation_leaks"]:
            raise FingerprintError("augmented copies leak across splits: " + "; ".join(checks["warnings"]))

        # record-level contamination (NLP / tabular): same normalised line in train and val/test
        record_stats: Dict[str, tuple] = {}
        if content_checks:
            lines_by_split: Dict[str, set] = defaultdict(set)
            for r in rows:
                if r.get("_lines") and r["split"] != "shared":
                    lines_by_split[r["split"]] |= r["_lines"]
            train_lines = lines_by_split.get("train", set())
            overlap = {sp: len(train_lines & v) for sp, v in lines_by_split.items() if sp != "train"}
            if train_lines:
                record_stats.update({sp: (len(v), overlap[sp]) for sp, v in lines_by_split.items() if sp != "train"})
            overlap = {k: v for k, v in overlap.items() if v}
            if overlap:
                checks["quality"]["record_overlap_with_train"] = overlap
                msg = ", ".join(f"{v} {k} record(s)" for k, v in overlap.items())
                checks["warnings"].append(f"{msg} also appear in train (exact, normalised) - test contamination")
                _log(f"dataset-fingerprint: WARNING {checks['warnings'][-1]}", quiet)
        nd_leaked: set = set()
        if nd_threshold is not None:
            ph_rows = [r for r in rows if (r.get("meta") or {}).get("dhash")]
            pairs, hot = perceptual.cross_split_pairs([(r["split"], int(r["meta"]["dhash"], 16)) for r in ph_rows], nd_threshold)
            sha_of = [r["sha256"] for r in ph_rows]
            pairs = [p for p in pairs if sha_of[p[0]] != sha_of[p[1]]]  # byte-identical ones are already reported
            nd = perceptual.summarize([r["source_path"] for r in ph_rows], [r["split"] for r in ph_rows], pairs)
            for i, j, _d in pairs:
                a, b = ph_rows[i], ph_rows[j]
                for x, y in ((a, b), (b, a)):
                    if y["split"] == "train" and x["split"] not in ("train", "shared"):
                        nd_leaked.add(x["source_path"])
            nd.update(threshold=nd_threshold, hashed=len(ph_rows), skipped_flat=hot)
            checks["quality"]["near_duplicates"] = nd
            if nd["with_train_match"]:
                msg = ", ".join(f"{v} {k}" for k, v in nd["with_train_match"].items())
                ex = nd["examples"][0]
                checks["warnings"].append(f"{msg} image(s) have a near-duplicate in train (dHash <= {nd_threshold}), e.g. {ex[0]} ~ {ex[1]}")
                _log(f"dataset-fingerprint: WARNING {checks['warnings'][-1]}", quiet)
        if temporal and strategy.get("time_ranges", {}).get("train") and strategy["time_ranges"].get("test"):
            tr, te = strategy["time_ranges"]["train"], strategy["time_ranges"]["test"]
            if tr["end"] > te["start"]:
                checks["warnings"].append(
                    f"train ends {tr['end']} after test starts {te['start']} - overlapping windows (long recordings straddle the cutoff)"
                )
                _log(f"dataset-fingerprint: WARNING {checks['warnings'][-1]}", quiet)

        # leak rate ---------------------------------------------------------------------
        by_type: Dict[str, set] = {"augmented_copy": checks["aug_leaked"]}
        train_sha = {r["sha256"] for r in rows if r["split"] == "train"}
        by_type["exact_copy"] = {r["source_path"] for r in rows if r["split"] not in ("train", "shared") and r["sha256"] in train_sha}
        if nd_threshold is not None:
            by_type["near_duplicate"] = nd_leaked
        # a recording/sequence folder name present in train and in val/test (only possible with given splits).
        # Not counted in the rate: datasets often restart numbering per split, so a shared name is a hint, not proof.
        rests = [r["key_rest"] for r in rows]
        _seqs, _steps = sequence_dirs(rests), sequence_steps(rests)
        rec_keys = [auto_group_key(r["key_rest"], _seqs, "label" in r, 0, 100, _steps) for r in rows]
        rec_keys = [k if k.startswith(("rec:", "seq:", "src:")) else "" for k in rec_keys]
        train_rec = {k for r, k in zip(rows, rec_keys) if k and r["split"] == "train"}
        same_rec = sorted({k for r, k in zip(rows, rec_keys) if k and r["split"] not in ("train", "shared") and k in train_rec})
        if temporal and strategy.get("time_ranges", {}).get("train"):
            t_end = max(r["_t"] for r in rows if r["split"] == "train")
            by_type["temporal_overlap"] = {r["source_path"] for r in rows if r["split"] not in ("train", "shared") and r["_t"] <= t_end}
        leakage = _leak_metrics(rows, by_type, record_stats)
        if same_rec:
            leakage["recording_names_also_in_train"] = {"count": len(same_rec), "examples": [k[4:] for k in same_rec[:5]]}
            checks["warnings"].append(
                f"{len(same_rec)} recording/sequence/source name(s) appear in train and val/test, "
                f"e.g. {same_rec[0][4:]} - same subject, session or source clip?"
            )
            _log(f"dataset-fingerprint: WARNING {checks['warnings'][-1]}", quiet)
        checks["quality"]["leakage"] = leakage
        if leakage.get("near_duplicates_uninformative"):
            checks["warnings"].append(
                f"near-duplicate hash matches >{NEAR_DUP_UNINFORMATIVE:.0%} of {', '.join(leakage['near_duplicates_uninformative'])} - "
                "the 64-bit hash cannot tell these images apart (small, centred, low-detail images); "
                "treat near-duplicate counts here as noise and use embeddings instead"
            )
            _log(f"dataset-fingerprint: WARNING {checks['warnings'][-1]}", quiet)
        if leakage.get("leak_rate"):
            _log(
                f"dataset-fingerprint: leak rate {leakage['leak_rate']:.2%} ({leakage['leaked_files']:,} of "
                f"{leakage['evaluation_files']:,} val/test files share content or a recording with train)",
                quiet,
            )
        if max_leak_rate is not None and (leakage.get("leak_rate") or 0) > max_leak_rate:
            raise FingerprintError(
                f"leak rate {leakage['leak_rate']:.2%} exceeds max_leak_rate {max_leak_rate:.2%}: "
                + json.dumps({k: v["by_type"] for k, v in leakage["by_split"].items()})
            )

        index = [{k: v for k, v in r.items() if k not in ("_mtime", "_stamp", "key_rest", "_lines", "_t", "_annotation")} for r in rows]
        index.sort(key=lambda r: (r["split"], r["path"]))

        # 5. lineage + version number ------------------------------------------------
        prev = self.latest()
        d = compute_diff(prev.version if prev else None, prev.records() if prev else [], "(new)", index)
        if prev and d.unchanged and not allow_empty and not version:
            _log(f"dataset-fingerprint: no changes since {prev.version} - nothing to commit", quiet)
            return prev
        if version:
            semver = "%d.%d.%d" % parse_semver(version)
            bump_used = "explicit"
        else:
            bump_used = d.suggested_bump() if bump == "auto" else bump
            semver = bump_semver(prev.semver if prev else None, bump_used)
            if not prev:
                bump_used = "initial"
        folder = self._folder(semver)
        full_version = folder
        d.target = full_version
        if self.loc.exists(f"{folder}/{MANIFEST}"):
            raise FingerprintError(f"{full_version} already exists - versions are immutable")
        for w in d.warnings():
            _log(f"dataset-fingerprint: WARNING {w}", quiet)
        if strategy.get("ratios"):
            present = {r["split"] for r in index}
            for s in strategy["ratios"]:
                if s not in present:
                    _log(f"dataset-fingerprint: WARNING split '{s}' is empty - too few files/groups for the requested ratios", quiet)

        data_loc = src if mode == "reference" else (self.loc if mode == "cas" else self.loc.child(folder))
        manifest = self._build_manifest(
            index,
            full_version,
            semver,
            message,
            src,
            data_loc,
            mode,
            transformations,
            source_batches,
            strategy,
            prev,
            d,
            bump_used,
            extra,
            f"column:{label_column}" if label_column else ("yolo:objects" if label_format == "yolo" else None),
        )
        manifest["lineage"]["parents"] = parent_locks
        if row_split:
            if "column" not in row_split:
                raise FingerprintError("row_split needs a 'column' plus 'val_from' and/or 'test_from'")
            manifest["row_split"] = {k: (str(v) if k != "column" else v) for k, v in row_split.items()}
        manifest["quality"] = checks["quality"]
        manifest["lineage"]["warnings"] = manifest["lineage"]["warnings"] + checks["warnings"]
        if dry_run:
            _log(f"dataset-fingerprint: dry run - would create {full_version}", quiet)
            return DatasetVersion(self, folder, manifest)

        # 6. write: data first, manifest last (a version without manifest doesn't exist)
        claim = self._claim(folder, manifest["commit_id"])
        if mode == "cas":
            prog = _Progress(len(index), "storing objects", quiet)
            seen: set = set()
            stored = [0]

            def put_obj(r: Dict[str, Any]) -> None:
                if r["path"] not in seen:
                    seen.add(r["path"])
                    if not self.loc.exists(r["path"]):
                        copy_file(src, r["source_path"], self.loc, r["path"], "copy")
                        stored[0] += 1
                prog.tick()

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                list(ex.map(put_obj, index))
            manifest["source"]["objects_written"] = stored[0]
            manifest["source"]["objects_reused"] = len({r["path"] for r in index}) - stored[0]
        if mode in ("copy", "hardlink"):
            dst = self.loc.child(folder)
            prog = _Progress(len(index), "copying", quiet)

            def cp(r: Dict[str, Any]) -> None:
                copy_file(src, r["source_path"], dst, r["path"], mode)
                prog.tick()

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                list(ex.map(cp, index))

        _write_index(self.loc, f"{folder}/{INDEX}", index)
        for s in manifest["metadata"]["splits"]:
            self.loc.write_text(f"{folder}/splits/{s}.txt", "".join(data_loc.uri(r["path"]) + "\n" for r in index if r["split"] == s))
        try:
            self.loc.refresh()
            if self.loc.exists(f"{folder}/{MANIFEST}"):
                raise FingerprintError(f"{full_version} was committed concurrently by another process")
            self.loc.write_text(f"{folder}/{MANIFEST}", json.dumps(manifest, indent=2, sort_keys=False) + "\n")
            self.loc.refresh()
            written = json.loads(self.loc.read_text(f"{folder}/{MANIFEST}"))
        except FingerprintError:
            raise
        except Exception as e:
            raise FingerprintError(f"{full_version}: concurrent commit conflict ({type(e).__name__})") from e
        self._release(claim)
        if written.get("commit_id") != manifest["commit_id"]:
            raise FingerprintError(f"{full_version} was committed concurrently by another process")
        self.loc.write_text(LEDGER_META, json.dumps({"name": self.name, "schema": SCHEMA, "latest": full_version}, indent=2) + "\n")
        _log(f"dataset-fingerprint: committed {full_version} ({bump_used}) fingerprint {manifest['fingerprint'][:23]}…", quiet)
        return DatasetVersion(self, folder, manifest)

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _assign_labels(rows: List[Dict[str, Any]], label_from: LabelFrom) -> None:
        if label_from in (None, "none"):
            return
        if callable(label_from):
            for r in rows:
                lab = label_from(r["source_path"])
                if lab is not None:
                    r["label"] = str(lab)
            return

        def parent(r: Dict[str, Any]) -> Optional[str]:
            parts = r.get("key_rest", r["rest"]).split("/")
            return parts[-2] if len(parts) >= 2 else None

        if label_from == "parent":
            for r in rows:
                p = parent(r)
                if p:
                    r["label"] = p
            return
        # auto: ImageFolder-style only when it really looks like class folders
        by_mod: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r.get("split") != "shared":
                by_mod[r["modality"]].append(r)
        for mod_rows in by_mod.values():
            parents: List[Optional[str]] = [parent(r) for r in mod_rows]
            if not mod_rows or any(p is None for p in parents):
                continue
            distinct: set = {p for p in parents if p}
            if not (2 <= len(distinct) <= max(2, min(1000, len(mod_rows) // 2))):
                continue
            if distinct & _NON_LABEL_DIRS or any(sensor_keyword(p) for p in distinct):
                continue
            # recordings / sequences (drive_012, seq_3, 2026-05-08) are not classes;
            # pure numbers (MNIST-style '0'..'9') are
            if sum(1 for p in distinct if re.search(r"[A-Za-z_\-].*\d+$", p) or re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", p)) > len(distinct) / 2:
                continue
            for r, p in zip(mod_rows, parents):
                r["label"] = p

    @staticmethod
    def _group_keys(rows: List[Dict[str, Any]], group_by: GroupBy, quiet: bool = False):
        if group_by == "file":
            return [r["key_rest"] for r in rows], "file"
        if group_by not in ("auto", None):
            return [group_key(r["key_rest"], group_by) for r in rows], _gb_repr(group_by)
        if group_by is None:
            return [r["key_rest"] for r in rows], "file"
        seqs = sequence_dirs([r["key_rest"] for r in rows])
        steps = sequence_steps([r["key_rest"] for r in rows])
        need = min(20, max(1, len(rows) // 2))
        plan = [(0, 100, "auto:recording"), (1, 100, "auto:sequence"), (1, 50, "auto:sequence"), (1, 25, "auto:sequence"), (1, 10, "auto:sequence")]
        best = None
        for level, block, name in plan:
            keys = [auto_group_key(r["key_rest"], seqs, "label" in r, level, block, steps) for r in rows]
            n = len(set(keys))
            label = name if block == 100 else f"{name}(block={block})"
            if n >= need:
                return keys, label
            if best is None or n > best[2]:
                best = (keys, label, n)
        # Not enough groups to hit the ratios exactly. Leakage beats balance: keep the
        # safest grouping that still has a few groups; only go per-file if there is none.
        if best and best[2] >= 3 and best[2] < len(rows):
            _log(
                f"dataset-fingerprint: WARNING only {best[2]} independent groups ({best[1]}); split sizes will be "
                "uneven but no recording/sequence is split apart",
                quiet,
            )
            return best[0], best[1]
        return [auto_group_key(r["key_rest"], seqs, "label" in r, 2) for r in rows], "auto:file"

    def _build_manifest(
        self,
        index,
        full_version,
        semver,
        message,
        src,
        data_loc,
        mode,
        transformations,
        source_batches,
        strategy,
        prev,
        d,
        bump_used,
        extra,
        label_column,
    ) -> Dict[str, Any]:
        total_bytes = sum(r["size"] for r in index)
        by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in index:
            by_split[r["split"]].append(r)
        order = ["train", "val", "test"]
        split_names = sorted(by_split, key=lambda s: (order.index(s) if s in order else 99, s))

        def modal_stats(rows):
            out: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                m = out.setdefault(r["modality"], {"files": 0, "bytes": 0})
                m["files"] += 1
                m["bytes"] += r["size"]
                meta = r.get("meta") or {}
                if meta.get("duration_sec"):
                    m["duration_sec"] = round(m.get("duration_sec", 0) + meta["duration_sec"], 3)
                if meta.get("records"):
                    m["records"] = m.get("records", 0) + meta["records"]
                if meta.get("width") and r["modality"] in ("image", "video"):
                    res = m.setdefault("resolutions", Counter())
                    res[f"{meta['width']}x{meta['height']}"] += 1
            for m in out.values():
                if "duration_sec" in m:
                    m["duration_hours"] = round(m["duration_sec"] / 3600, 4)
                if "resolutions" in m:
                    m["resolutions"] = dict(m["resolutions"].most_common(10))
            return dict(sorted(out.items()))

        def labels(rows) -> Counter:
            c: Counter = Counter()
            for r in rows:
                rec = (r.get("meta") or {}).get("label_counts")
                if rec:
                    c.update(rec)
                elif r.get("label") is not None:
                    c[r["label"]] += 1
            return c

        splits_meta = {}
        for s in split_names:
            rows = by_split[s]
            lab = labels(rows)
            splits_meta[s] = {
                "files": len(rows),
                "bytes": sum(r["size"] for r in rows),
                "fraction": round(len(rows) / len(index), 4),
                "modalities": {k: v["files"] for k, v in modal_stats(rows).items()},
            }
            if lab:
                splits_meta[s]["class_counts"] = dict(sorted(lab.items()))

        all_labels = labels(index)
        n_lab = sum(all_labels.values())
        seqs = sequence_dirs([r["source_path"] for r in index])
        seq_stats: Dict[str, Dict[str, Any]] = {}
        for d_, n in seqs.items():
            mods = Counter(r["modality"] for r in index if r["source_path"].rsplit("/", 1)[0] == d_)
            mod = mods.most_common(1)[0][0] if mods else "other"
            st = seq_stats.setdefault(mod, {"sequences": 0, "frames": 0, "min_frames": n, "max_frames": n})
            st["sequences"] += 1
            st["frames"] += n
            st["min_frames"] = min(st["min_frames"], n)
            st["max_frames"] = max(st["max_frames"], n)
        checksums = {"algorithm": ALGORITHM}
        for s in split_names:
            checksums[f"{s}_split"] = digest((r["path"], r["sha256"]) for r in by_split[s])
        fingerprint = digest((r["source_path"] + "|" + r["split"], r["sha256"]) for r in index)
        checksums["dataset"] = fingerprint

        batches = list(source_batches) if source_batches else sorted({r["source_path"].split("/")[0] for r in index if "/" in r["source_path"]})[:200]

        manifest: Dict[str, Any] = {
            "schema": SCHEMA,
            "name": self.name,
            "version": full_version,
            "semver": semver,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "created_by": _who(),
            "commit_id": uuid.uuid4().hex,
            "message": message,
            "source": {"uri": src.uri(), "mode": mode, "batches": batches},
            "data_location": data_loc.uri(),
            "transformations": list(transformations),
            "split_strategy": strategy,
            "metadata": {
                "total_files": len(index),
                "total_bytes": total_bytes,
                "total_size_gb": round(total_bytes / 1024**3, 4),
                "modalities": modal_stats(index),
                "splits": splits_meta,
                "split": {s: v["fraction"] for s, v in splits_meta.items()},
            },
            "checksums": checksums,
            "fingerprint": fingerprint,
            "lineage": {
                "previous_version": prev.version if prev else None,
                "previous_fingerprint": prev.fingerprint if prev else None,
                "reason_for_update": message,
                "bump": bump_used,
                "changes": d.summary(),
                "warnings": d.warnings(),
            },
        }
        if seq_stats:
            manifest["metadata"]["sequences"] = seq_stats
        starts = [(r["meta"]["start"], r["meta"].get("duration_sec") or 0) for r in index if (r.get("meta") or {}).get("start") is not None]
        if starts:
            t0 = min(a for a, _ in starts)
            t1 = max(a + b for a, b in starts)
            manifest["metadata"]["temporal_range"] = {
                "start": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"),
                "end": datetime.fromtimestamp(t1, timezone.utc).isoformat(timespec="seconds"),
            }
        sensors = sorted({r["modality"] for r in index} - {"text", "tabular", "array", "archive", "other", "log"})
        manifest["metadata"]["sensors"] = sensors
        if all_labels:
            manifest["metadata"]["num_classes"] = len(all_labels)
            manifest["metadata"]["label_source"] = (
                label_column if label_column and any((r.get("meta") or {}).get("label_counts") for r in index) else "path"
            )
            manifest["metadata"]["class_distribution"] = {k: round(v / n_lab, 4) for k, v in all_labels.most_common()}
        if extra:
            manifest["extra"] = extra
        return manifest

    # -- verify ------------------------------------------------------------------------
    def verify(
        self,
        version: Union[str, DatasetVersion, None] = "latest",
        sample: Optional[float] = None,
        workers: int = 8,
        quiet: bool = True,
        quick: bool = False,
    ) -> VerifyReport:
        """Re-hash files and compare with the index; also checks index vs manifest
        digests (catches someone editing index.jsonl.gz to hide a change)."""
        v = version if isinstance(version, DatasetVersion) else self.get(version)
        rows = v.records()
        rep = VerifyReport(v.version)
        by_split: Dict[str, List] = defaultdict(list)
        for r in rows:
            by_split[r["split"]].append((r["path"], r["sha256"]))
        for s, items in by_split.items():
            if v.manifest["checksums"].get(f"{s}_split") != digest(items):
                rep.index_tampered.append(s)
        if sample and 0 < sample < 1:
            rows = random.Random(0).sample(rows, max(1, int(len(rows) * sample)))  # sampling, not crypto  # nosec B311
        loc = v.data_location
        prog = _Progress(len(rows), "verifying", quiet)

        def check(r):
            try:
                if quick:  # existence + size only: seconds instead of hours on TB data
                    ok = loc.size(r["path"]) == r["size"]
                else:
                    with loc.open(r["path"], "rb") as f:
                        ok = hash_stream(f) == r["sha256"]
                res = "ok" if ok else "modified"
            except UnsafePathError:
                res = "unsafe"
            except (FileNotFoundError, OSError):
                res = "missing"
            prog.tick()
            return r["path"], res

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for path, res in ex.map(check, rows):
                rep.checked += 1
                if res == "ok":
                    rep.ok_files += 1
                elif res == "missing":
                    rep.missing.append(path)
                elif res == "unsafe":
                    rep.unsafe.append(path)
                else:
                    rep.modified.append(path)
        return rep


def _gb_repr(gb: GroupBy) -> Optional[str]:
    if gb is None or isinstance(gb, str):
        return gb
    return getattr(gb, "__name__", "callable")


# convenience functional API ------------------------------------------------------------
def record_training(
    model_version: str,
    datasets: Sequence[DatasetVersion],
    metrics: Optional[Dict[str, Any]] = None,
    recipe: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """One training run, many datasets: logs the run in every dataset's usage.jsonl and
    returns {'model_version', 'datasets': [locks], 'recipe', 'seed', ...} to store with the model."""
    locks = [d.lock() for d in datasets]
    for d in datasets:
        d.record_model(model_version, metrics, recipe=recipe, seed=seed, trained_with=locks, **extra)
    out = {"model_version": model_version, "datasets": locks, "metrics": metrics or {}, **extra}
    if recipe is not None:
        out["recipe"] = recipe
    if seed is not None:
        out["seed"] = seed
    return out


def commit(source: str, ledger: Optional[str] = None, name: Optional[str] = None, **kw: Any) -> DatasetVersion:
    """dataset_fingerprint.commit("./data", "s3://bucket/ledgers/cars", message="new batch")"""
    return Ledger(ledger, name=name, storage_options=kw.pop("storage_options", None)).commit(source, **kw)


def load(
    ledger: Optional[str] = None, version: str = "latest", name: Optional[str] = None, storage_options: Optional[Dict[str, Any]] = None
) -> DatasetVersion:
    """dataset_fingerprint.load("s3://bucket/ledgers/cars", "1.2.0").files("train")"""
    return Ledger(ledger, name=name, storage_options=storage_options).get(version)
