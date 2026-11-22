"""Version-to-version diffs and semantic-version bump inference."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)$")


def parse_semver(v: str) -> Tuple[int, int, int]:
    m = _SEMVER.search(v)
    if not m:
        raise ValueError(f"not a semantic version: {v!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump_semver(v: Optional[str], part: str) -> str:
    if v is None:
        return "1.0.0"
    major, minor, patch = parse_semver(v)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"bump must be major|minor|patch|auto, got {part!r}")


@dataclass
class Diff:
    base: Optional[str]
    target: str
    added: List[Dict[str, Any]] = field(default_factory=list)
    removed: List[Dict[str, Any]] = field(default_factory=list)
    modified: List[Dict[str, Any]] = field(default_factory=list)
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    split_changed: List[Tuple[str, str, str]] = field(default_factory=list)  # (path, old, new)
    modalities_added: List[str] = field(default_factory=list)
    modalities_removed: List[str] = field(default_factory=list)
    label_delta: Dict[str, int] = field(default_factory=dict)

    @property
    def unchanged(self) -> bool:
        return not (self.added or self.removed or self.modified or self.renamed or self.split_changed)

    def suggested_bump(self) -> str:
        """Article semantics: new modality / collection process -> major,
        new data -> minor, filtering / fixes -> patch."""
        if self.modalities_added or self.modalities_removed:
            return "major"
        if self.added:
            return "minor"
        return "patch"

    def summary(self) -> Dict[str, Any]:
        def by_split(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
            return dict(Counter(r["split"] for r in rows))

        out: Dict[str, Any] = {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "renamed": len(self.renamed),
            "split_changed": len(self.split_changed),
        }
        if self.added:
            out["added_by_split"] = by_split(self.added)
        if self.removed:
            out["removed_by_split"] = by_split(self.removed)
        if self.modalities_added:
            out["modalities_added"] = self.modalities_added
        if self.modalities_removed:
            out["modalities_removed"] = self.modalities_removed
        if self.label_delta:
            out["label_delta"] = self.label_delta
        return out

    def warnings(self) -> List[str]:
        w = []
        if self.split_changed:
            moved_to_train = sum(1 for _, o, n in self.split_changed if n == "train" and o != "train")
            w.append(
                f"{len(self.split_changed)} file(s) changed split since {self.base}"
                + (f" ({moved_to_train} moved into train - possible evaluation leakage)" if moved_to_train else "")
            )
        return w

    def render(self, limit: int = 10) -> str:
        lines = [f"{self.base or '(empty)'} -> {self.target}"]
        s = self.summary()
        lines.append(
            f"  +{s['added']} added  -{s['removed']} removed  ~{s['modified']} modified  ↻{s['renamed']} renamed  ⇄{s['split_changed']} split-changed"
        )
        for key, sym in (("added", "+"), ("removed", "-"), ("modified", "~")):
            rows = getattr(self, key)
            for r in rows[:limit]:
                lines.append(f"  {sym} [{r['split']}] {r['source_path']}")
            if len(rows) > limit:
                lines.append(f"  {sym} ... {len(rows) - limit} more")
        for a, b in self.renamed[:limit]:
            lines.append(f"  ↻ {a} -> {b}")
        for p, o, n in self.split_changed[:limit]:
            lines.append(f"  ⇄ {p}: {o} -> {n}")
        if self.modalities_added:
            lines.append(f"  new modalities: {', '.join(self.modalities_added)}")
        if self.modalities_removed:
            lines.append(f"  dropped modalities: {', '.join(self.modalities_removed)}")
        if self.label_delta:
            lines.append("  label delta: " + ", ".join(f"{k}:{v:+d}" for k, v in sorted(self.label_delta.items())))
        for w in self.warnings():
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def compute_diff(base: Optional[str], old: List[Dict[str, Any]], target: str, new: List[Dict[str, Any]]) -> Diff:
    d = Diff(base=base, target=target)
    o = {r["source_path"]: r for r in old}
    n = {r["source_path"]: r for r in new}
    added = [n[k] for k in n.keys() - o.keys()]
    removed = [o[k] for k in o.keys() - n.keys()]

    # renames: same content, different path
    removed_by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for r in removed:
        removed_by_hash.setdefault(r["sha256"], []).append(r)
    still_added = []
    for r in sorted(added, key=lambda x: x["source_path"]):
        cands = removed_by_hash.get(r["sha256"])
        if cands:
            old_r = cands.pop(0)
            d.renamed.append((old_r["source_path"], r["source_path"]))
        else:
            still_added.append(r)
    d.added = still_added
    d.removed = sorted((r for rs in removed_by_hash.values() for r in rs), key=lambda x: x["source_path"])

    for k in sorted(o.keys() & n.keys()):
        if o[k]["sha256"] != n[k]["sha256"]:
            d.modified.append(n[k])
        if o[k]["split"] != n[k]["split"]:
            d.split_changed.append((k, o[k]["split"], n[k]["split"]))

    om = {r["modality"] for r in old}
    nm = {r["modality"] for r in new}
    if old:
        d.modalities_added = sorted(nm - om)
        d.modalities_removed = sorted(om - nm)

    ol: Counter = Counter(str(r["label"]) for r in old if r.get("label") is not None)
    nl: Counter = Counter(str(r["label"]) for r in new if r.get("label") is not None)
    d.label_delta = {k: nl[k] - ol[k] for k in set(ol) | set(nl) if nl[k] != ol[k]}
    return d
