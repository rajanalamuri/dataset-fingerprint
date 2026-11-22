"""Anonymous leakage audit for datasets you cannot share.

Run it where the data lives (laptop, VM, Databricks, CI runner). It reads the datasets, runs
dataset-fingerprint's commit-time checks in a throw-away ledger, and writes ONLY aggregate numbers:
no paths, file names, class names, dataset names, warning texts or examples ever leave the run.
Review the output files before sharing them.

    pip install "dataset-fingerprint[image]"          # + [s3] / [azure] / [gcs] for object stores
    python anonymous_audit.py datasets.json --out audit

datasets.json (stays with you; nothing in it is copied to the output except the `domain` label):

    [
      {"source": "/data/camera/front_v3",      "domain": "vision"},
      {"source": "s3://bucket/radar/2026-q3/", "domain": "radar"},
      {"source": "/dbfs/mnt/audio/kws",        "domain": "audio", "split": "0.8,0.1,0.1"},
      {"source": "/data/det/yolo",             "domain": "vision-detection", "label_format": "yolo"}
    ]

Per entry, any dataset_fingerprint commit option may be added (split, group_by, label_format,
time_cutoffs, include, exclude, ...). `domain` should be a generic word (vision, audio, radar,
lidar, imu, can, text, tabular, ...): it is the only free text copied to the output.

Outputs: <out>.json (per-dataset numbers + totals) and <out>.md (tables for the paper).
Small counts are reported as "<5" and the rest rounded unless --exact is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dataset_fingerprint as fp

SAFE_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9 _/+.-]{0,30}$")
PASS_THROUGH = {
    "split", "group_by", "label_format", "class_names", "time_cutoffs", "row_split", "include", "exclude",
    "probe", "modality_rules", "label_from", "label_column", "near_duplicates", "source_storage_options", "workers",
}  # fmt: skip


def size_bucket(n: int) -> str:
    if n < 1_000:
        return "<1k"
    for lim, name in ((10_000, "1k-10k"), (100_000, "10k-100k"), (1_000_000, "100k-1M"), (10_000_000, "1M-10M")):
        if n < lim:
            return name
    return ">10M"


class Anon:
    """Rounds counts so no single file or small group can be singled out."""

    def __init__(self, exact: bool):
        self.exact = exact

    def n(self, x: Optional[int]) -> Any:
        if x is None or self.exact:
            return x
        if x == 0:
            return 0
        if x < 5:
            return "<5"
        mag = 10 ** max(0, int(math.log10(x)) - 1)  # two significant figures
        return int(round(x / mag) * mag)

    def r(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return x if self.exact else float(f"{x:.2g}")


def audit_one(i: int, spec: Dict[str, Any], anon: Anon, near_dups: int) -> Dict[str, Any]:
    domain = str(spec.get("domain", "unspecified")).strip().lower()
    if not SAFE_DOMAIN.match(domain):
        raise SystemExit(f"dataset #{i + 1}: domain must be a short generic word (got {domain!r})")
    opts = {k: v for k, v in spec.items() if k in PASS_THROUGH}
    unknown = set(spec) - PASS_THROUGH - {"source", "domain"}
    if unknown:
        raise SystemExit(f"dataset #{i + 1}: unknown option(s) {sorted(unknown)}")
    opts.setdefault("near_duplicates", near_dups if near_dups >= 0 else False)

    work = Path(tempfile.mkdtemp(prefix="fp-audit-"))
    os.environ["FINGERPRINT_CACHE"] = str(work / "cache.sqlite")
    t = time.time()
    try:
        v = fp.Ledger(str(work / "ledger")).commit(spec["source"], quiet=True, cache=True, **opts)
        m = v.manifest
        recs = v.records()
    except Exception as e:  # report locally, share only the error type
        print(f"    D{i + 1:02d} failed locally: {e}", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        return {"id": f"D{i + 1:02d}", "domain": domain, "error": type(e).__name__}
    finally:
        os.environ.pop("FINGERPRINT_CACHE", None)
    dt = time.time() - t
    shutil.rmtree(work, ignore_errors=True)  # the throw-away ledger and its index never leave this machine

    md, q, st = m["metadata"], m["quality"], m["split_strategy"]
    lk = q.get("leakage") or {}
    nd = q.get("near_duplicates") or {}
    files = int(md["total_files"])
    split_sizes = {}
    for r in recs:
        split_sizes[r["split"]] = split_sizes.get(r["split"], 0) + 1
    by_type: Dict[str, int] = {}
    rec_total = rec_leak = 0
    for sp in (lk.get("by_split") or {}).values():
        for k, c in sp.get("by_type", {}).items():
            by_type[k] = by_type.get(k, 0) + c
        rec_total += sp.get("records", 0)
        rec_leak += sp.get("records_also_in_train", 0)
    names = lk.get("recording_names_also_in_train") or {}
    temporal_overlap = any("overlapping windows" in w for w in m["lineage"].get("warnings", []))
    return {
        "id": f"D{i + 1:02d}",
        "domain": domain,
        "files": anon.n(files),
        "size": size_bucket(files),
        "modalities": {k: anon.n(v["files"] if isinstance(v, dict) else v) for k, v in (md.get("modalities") or {}).items()},
        "split_type": st.get("type"),
        "grouping": st.get("group_by"),
        "groups": anon.n(st.get("groups")),
        "splits": {k: anon.n(c) for k, c in sorted(split_sizes.items())},
        "leak_rate": anon.r(lk.get("leak_rate")),
        "leaked_files": anon.n(lk.get("leaked_files")),
        "leaked_by_type": {k: anon.n(c) for k, c in sorted(by_type.items()) if k != "near_duplicate"},
        "suspected_rate": anon.r(lk.get("suspected_rate")) if nd.get("hashed") else None,
        "near_duplicate_files": anon.n(by_type.get("near_duplicate", 0)) if nd.get("hashed") else None,
        "images_hashed": anon.n(nd.get("hashed")) if nd else None,
        "near_duplicate_hash_uninformative": bool(lk.get("near_duplicates_uninformative")),
        "recording_names_in_train_and_eval": anon.n(names.get("count", 0)),
        "record_leak_rate": anon.r(rec_leak / rec_total) if rec_total else None,
        "temporal_overlap": temporal_overlap,
        "duplicate_files_within": anon.n(q.get("duplicate_files")),
        "augmented_files": anon.n(q.get("augmented_files")),
        "augmented_in_eval": anon.n(q.get("augmented_in_eval")),
        "empty_or_unreadable": anon.n((q.get("empty_files") or 0) + (q.get("unreadable_files") or 0)),
        "audit_seconds": round(dt) if anon.exact else int(round(dt, -1)),
    }


def any_leak(d: Dict[str, Any]) -> bool:
    return (
        bool(d.get("leak_rate"))
        or bool(d.get("recording_names_in_train_and_eval"))
        or bool(d.get("temporal_overlap"))
        or bool(d.get("record_leak_rate"))
    )


def markdown(rows: List[Dict[str, Any]], totals: Dict[str, Any]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.2%}"

    out = ["# Anonymous leakage audit", "", f"{totals['datasets']} datasets, {totals['with_any_leak_signal']} with at least one leak signal.", ""]
    out += [
        "| ID | Domain | Size | Splits | Leak rate | Leaks by type | Suspected | Recording names in train+eval | Record leak | Temporal overlap |",
        "|---|---|---|---|---:|---|---:|---:|---:|:-:|",
    ]
    for d in rows:
        if "error" in d:
            out.append(f"| {d['id']} | {d['domain']} | – | could not be read ({d['error']}) | | | | | | |")
            continue
        types = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d["leaked_by_type"].items()) or "–"
        sus = pct(d["suspected_rate"]) + (" (hash uninformative)" if d["near_duplicate_hash_uninformative"] else "")
        out.append(
            f"| {d['id']} | {d['domain']} | {d['size']} | {d['split_type']} ({len(d['splits'])}) | {pct(d['leak_rate'])} | {types} | {sus} | "
            f"{d['recording_names_in_train_and_eval']} | {pct(d['record_leak_rate'])} | {'yes' if d['temporal_overlap'] else 'no'} |"
        )
    out += ["", "Totals: " + json.dumps({k: v for k, v in totals.items() if k != "env"}), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", help="JSON list of {source, domain, ...options}")
    ap.add_argument("--out", default="audit", help="output prefix (writes <out>.json and <out>.md)")
    ap.add_argument("--near-dups", type=int, default=4, help="perceptual threshold for images; -1 to skip (default 4)")
    ap.add_argument("--exact", action="store_true", help="report exact counts (default: '<5' and 2 significant figures)")
    a = ap.parse_args()
    specs = json.loads(Path(a.datasets).read_text())
    if not isinstance(specs, list) or not all(isinstance(s, dict) and "source" in s for s in specs):
        raise SystemExit("datasets file must be a JSON list of objects with a 'source'")
    anon = Anon(a.exact)
    rows = []
    for i, s in enumerate(specs):
        print(f"[{i + 1}/{len(specs)}] auditing D{i + 1:02d} ({s.get('domain', 'unspecified')}) ...", flush=True)
        rows.append(audit_one(i, s, anon, a.near_dups))
    ok = [d for d in rows if "error" not in d]
    rates = [d["leak_rate"] for d in ok if d["leak_rate"] is not None]
    totals = {
        "datasets": len(rows),
        "audited": len(ok),
        "with_any_leak_signal": sum(1 for d in ok if any_leak(d)),
        "with_confirmed_leaks": sum(1 for d in ok if d["leak_rate"]),
        "with_recording_names_in_train_and_eval": sum(1 for d in ok if d["recording_names_in_train_and_eval"]),
        "with_temporal_overlap": sum(1 for d in ok if d["temporal_overlap"]),
        "median_leak_rate": anon.r(statistics.median(rates)) if rates else None,
        "max_leak_rate": anon.r(max(rates)) if rates else None,
        "domains": sorted({d["domain"] for d in rows}),
        "env": {"dataset_fingerprint": fp.__version__, "python": platform.python_version(), "exact_counts": a.exact},
    }
    Path(f"{a.out}.json").write_text(json.dumps({"datasets": rows, "totals": totals}, indent=2))
    Path(f"{a.out}.md").write_text(markdown(rows, totals))
    print(f"\nwrote {a.out}.json and {a.out}.md: review both before sharing them.")


if __name__ == "__main__":
    main()
