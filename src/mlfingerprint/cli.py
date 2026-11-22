"""mlfingerprint command line. Set FINGERPRINT_ROOT to skip --ledger everywhere."""

from __future__ import annotations

import argparse
import os
import json
import sys
from typing import List, Optional

from . import __version__
from .ledger import FingerprintError, Ledger


def _ledger(a) -> Ledger:
    opts = json.loads(a.storage_options) if getattr(a, "storage_options", None) else None
    return Ledger(a.ledger, name=a.name, storage_options=opts)


def _rules(specs):
    out = {}
    for spec in specs or []:
        name, _, pat = spec.partition("=")
        if not pat:
            raise FingerprintError(f"--modality expects NAME=GLOB, got {spec!r}")
        out.setdefault(name.strip(), []).append(pat.strip())
    return out or None


def _kv(specs):
    """['val=2026-05-01', 'test=2026-05-15'] -> dict"""
    out = {}
    for spec in specs or []:
        for part in spec.split(","):
            k, _, v = part.partition("=")
            if not v:
                raise FingerprintError(f"expected KEY=VALUE, got {part!r}")
            out[k.strip()] = v.strip()
    return out or None


def _names(spec):
    """'person,bicycle,...' or a path to a file with one class name per line."""
    if not spec:
        return None
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    return [x.strip() for x in spec.split(",")]


def cmd_commit(a) -> int:
    lg = _ledger(a)
    v = lg.commit(
        a.source,
        message=a.message or "",
        version=a.version,
        bump=a.bump,
        transformations=a.transform or [],
        source_batches=a.source_batch,
        mode=a.mode,
        split=a.split,
        group_by=a.group_by,
        seed=a.seed,
        modality_rules=_rules(a.modality),
        label_from=a.label_from,
        label_column=a.label_column,
        label_format=a.label_format,
        class_names=_names(a.class_names),
        probe=a.probe,
        include=a.include,
        exclude=a.exclude,
        workers=a.workers,
        cache=not a.no_cache,
        allow_empty=a.allow_empty,
        dry_run=a.dry_run,
        quiet=a.quiet,
        parents=a.parent,
        augmentation=a.augmentation,
        content_checks=not a.no_content_checks,
        near_duplicates=False if a.near_dups is None else a.near_dups,
        time_cutoffs=_kv(a.time_cutoff),
        row_split=_kv(a.row_split),
        extra=json.loads(a.extra) if a.extra else None,
    )
    print(json.dumps(v.manifest, indent=2) if a.json else v.summary())
    return 0


def cmd_log(a) -> int:
    lg = _ledger(a)
    vs = lg.log()
    if a.json:
        print(json.dumps([v.manifest for v in vs], indent=2))
        return 0
    if not vs:
        print("(no versions)")
        return 0
    print(f"{'VERSION':<28}{'CREATED':<27}{'FILES':>9}  {'BUMP':<9}MESSAGE")
    for v in reversed(vs):
        m = v.manifest
        print(
            f"{m['version']:<28}{m['created_at']:<27}{m['metadata']['total_files']:>9,}  "
            f"{(m.get('lineage') or {}).get('bump', ''):<9}{m.get('message', '')}"
        )
    return 0


def cmd_show(a) -> int:
    v = _ledger(a).get(a.version)
    print(json.dumps(v.manifest, indent=2) if a.json else v.summary())
    if not a.json:
        for u in v.usage():
            print(f"  used by     : {u['model_version']}  {json.dumps(u.get('metrics') or {})}")
    return 0


def cmd_verify(a) -> int:
    rep = _ledger(a).verify(a.version, sample=a.sample, workers=a.workers, quiet=a.quiet, quick=a.quick)
    if a.json:
        print(json.dumps(rep.__dict__ | {"ok": rep.ok}, indent=2))
    else:
        print(rep.render())
    return 0 if rep.ok else 1


def cmd_diff(a) -> int:
    d = _ledger(a).diff(a.base, a.target)
    print(json.dumps(d.summary(), indent=2) if a.json else d.render(limit=a.limit))
    return 0


def cmd_files(a) -> int:
    v = _ledger(a).get(a.version)
    for f in v.files(a.split):
        print(f)
    return 0


def cmd_tag(a) -> int:
    lg = _ledger(a)
    if a.delete:
        lg.untag(a.tag)
    elif a.version_or_tag:
        e = lg.tag(a.version_or_tag, a.tag, a.message or "")
        print(f"{a.tag} -> {e['version']}")
    return 0


def cmd_tags(a) -> int:
    tags = _ledger(a).tags()
    if a.json:
        print(json.dumps(tags, indent=2))
    for name, e in sorted(tags.items()):
        if not a.json:
            print(f"{name:<20}{e['version']:<32}{e['updated_at']}  {e.get('message', '')}")
    return 0


def cmd_lineage(a) -> int:
    v = _ledger(a).get(a.version)
    tree = v.trace()
    if a.json:
        print(json.dumps(tree, indent=2))
    else:

        def show(n, depth=0):
            label = n.get("dataset_version") or json.dumps({k: n[k] for k in n if k != "parents"})
            print("  " * depth + ("└─ " if depth else "") + label + (f"  ({n['message']})" if n.get("message") else ""))
            for p in n.get("parents", []):
                show(p, depth + 1)

        show(tree)
    warnings = v.check_lineage()
    for w in warnings:
        print(f"! {w}", file=sys.stderr)
    return 1 if any(not w.startswith("stale") for w in warnings) else 0


def cmd_checkout(a) -> int:
    print(_ledger(a).get(a.version).checkout(a.dest, split=a.split, mode=a.mode))
    return 0


def cmd_lock(a) -> int:
    print(json.dumps(_ledger(a).get(a.version).lock(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fingerprint", description="Manifest-based dataset versioning for any modality.")
    p.add_argument("--version", action="version", version=f"mlfingerprint {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", "-L", help="ledger root (local, s3://, az://, abfss://, gs://, dbfs:/, /Volumes/...). Default: $FINGERPRINT_ROOT")
    common.add_argument("--name", "-n", help="dataset name (default: from ledger.json or 'dataset')")
    common.add_argument("--storage-options", help='JSON fsspec options, e.g. \'{"account_name":"x"}\'')
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--quiet", "-q", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("commit", parents=[common], help="version a dataset folder")
    c.add_argument("source")
    c.add_argument("-m", "--message")
    c.add_argument("--version", dest="version", help="explicit semver, e.g. 2.0.0")
    c.add_argument("--bump", default="auto", choices=["auto", "major", "minor", "patch"])
    c.add_argument("-t", "--transform", action="append", help="transformation applied (repeatable)")
    c.add_argument("--source-batch", action="append", help="raw source batch (repeatable)")
    c.add_argument("--mode", default="reference", choices=["reference", "copy", "hardlink", "cas"])
    c.add_argument("--split", default="auto", help="auto | none | 0.8,0.1,0.1 | train=0.9,test=0.1 | temporal | temporal:0.7,0.15,0.15")
    c.add_argument("--time-cutoff", action="append", help="temporal cutoffs, e.g. val=2026-05-01,test=2026-05-15")
    c.add_argument(
        "--row-split", action="append", help="row-level time split for one series file: column=timestamp,val_from=2026-05-01,test_from=2026-05-15"
    )
    c.add_argument("--parent", action="append", help="derived from: ROOT@VERSION (repeatable)")
    c.add_argument("--augmentation", default="warn", choices=["warn", "error", "ignore"])
    c.add_argument("--no-content-checks", action="store_true", help="skip record-level train/test overlap check")
    c.add_argument(
        "--near-dups",
        nargs="?",
        type=int,
        const=4,
        default=None,
        metavar="T",
        help="flag re-encoded/resized copies of the same image across splits (dHash Hamming <= T, default 4; needs Pillow)",
    )
    c.add_argument("--extra", help="JSON object stored in the manifest")
    c.add_argument("--group-by", default="auto", help="auto (recording/sequence-safe) | file | parent | stem | regex - what stays in one split")
    c.add_argument("--modality", action="append", help="NAME=GLOB modality rule, e.g. radar='*/mmwave/**' (repeatable)")
    c.add_argument("--seed", default="mlfingerprint")
    c.add_argument("--label-from", default="auto", choices=["auto", "parent", "none"])
    c.add_argument("--label-column", help="label column inside csv/tsv/jsonl files")
    c.add_argument("--label-format", choices=["yolo"], help="count objects per class in YOLO .txt label files")
    c.add_argument("--class-names", help="class names for --label-format: comma list or a file with one per line")
    c.add_argument("--probe", action="store_true", help="read durations, resolutions, record counts")
    c.add_argument("--include", action="append", help="glob, repeatable")
    c.add_argument("--exclude", action="append", help="glob, repeatable")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--no-cache", action="store_true")
    c.add_argument("--allow-empty", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_commit)

    s = sub.add_parser("log", parents=[common], help="list versions")
    s.set_defaults(func=cmd_log)
    s = sub.add_parser("show", parents=[common], help="describe a version")
    s.add_argument("version", nargs="?", default="latest")
    s.set_defaults(func=cmd_show)
    s = sub.add_parser("verify", parents=[common], help="re-hash and check integrity (exit 1 on failure)")
    s.add_argument("version", nargs="?", default="latest")
    s.add_argument("--sample", type=float, help="verify a random fraction, e.g. 0.05")
    s.add_argument("--quick", action="store_true", help="existence + size only (no re-hashing)")
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(func=cmd_verify)
    s = sub.add_parser("diff", parents=[common], help="compare two versions")
    s.add_argument("base")
    s.add_argument("target", nargs="?", default="latest")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_diff)
    s = sub.add_parser("files", parents=[common], help="print file URIs of a version/split")
    s.add_argument("version", nargs="?", default="latest")
    s.add_argument("--split")
    s.set_defaults(func=cmd_files)
    s = sub.add_parser("tag", parents=[common], help="name a version (prod, golden, ...); -d to remove")
    s.add_argument("tag")
    s.add_argument("version_or_tag", nargs="?")
    s.add_argument("-m", "--message")
    s.add_argument("-d", "--delete", action="store_true")
    s.set_defaults(func=cmd_tag)
    s = sub.add_parser("tags", parents=[common], help="list tags")
    s.set_defaults(func=cmd_tags)
    s = sub.add_parser("lineage", parents=[common], help="ancestry tree + stale/altered parent check")
    s.add_argument("version", nargs="?", default="latest")
    s.set_defaults(func=cmd_lineage)
    s = sub.add_parser("checkout", parents=[common], help="materialise a version as DEST/<split>/...")
    s.add_argument("version")
    s.add_argument("dest")
    s.add_argument("--split")
    s.add_argument("--mode", default="copy", choices=["copy", "hardlink", "symlink"])
    s.set_defaults(func=cmd_checkout)
    s = sub.add_parser("lock", parents=[common], help="print the lock dict to store with a model")
    s.add_argument("version", nargs="?", default="latest")
    s.set_defaults(func=cmd_lock)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except FingerprintError as e:
        print(f"mlfingerprint: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
