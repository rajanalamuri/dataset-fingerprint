"""Security regression tests.

Threat model: anyone who can write to a ledger can try to make *readers* misbehave
(write outside a checkout folder, read files outside the dataset, hide tampering).
Write access to the data itself is detected by fingerprints; write access to the
ledger is detected by the lock each model keeps (see SECURITY.md).
"""

from __future__ import annotations

import ast
import gzip
import json
import os
from pathlib import Path

import pytest

import dataset_fingerprint as fp
from dataset_fingerprint.storage import Location, UnsafePathError, safe_rel

pytestmark = pytest.mark.security
SRC = Path(fp.__file__).parent


def put(p: Path, data: str | bytes):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode() if isinstance(data, str) else data)


@pytest.fixture
def ds(tmp_path):
    src = tmp_path / "src"
    for split in ("train", "test"):
        for i in range(4):
            put(src / split / f"f{i}.txt", f"{split}{i}")
    return src


def _rewrite_index(ledger_dir: Path, mutate):
    idx = next(ledger_dir.glob("*/index.jsonl.gz"))
    rows = [json.loads(x) for x in gzip.decompress(idx.read_bytes()).decode().splitlines()]
    mutate(rows)
    idx.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))


# ------------------------------------------------------------------ path traversal
@pytest.mark.parametrize("bad", ["../etc/passwd", "a/../../x", "/etc/passwd", "C:/Windows/x", "a\\..\\b", "a\x00b"])
def test_safe_rel_rejects_escapes(bad):
    with pytest.raises(UnsafePathError):
        safe_rel(bad)


@pytest.mark.parametrize("ok", ["a.txt", "train/cat/1.jpg", "objects/ab/abcdef.png", ".claims/x", "a..b/c"])
def test_safe_rel_allows_normal_paths(ok):
    assert safe_rel(ok) == ok


def test_location_refuses_traversal(tmp_path):
    loc = Location(str(tmp_path / "root"))
    for bad in ("../outside.txt", "/tmp/x"):
        with pytest.raises(UnsafePathError):
            loc.open(bad, "wb")
    assert not (tmp_path / "outside.txt").exists()


def test_checkout_with_malicious_index_writes_nothing_outside(tmp_path, ds):
    L = tmp_path / "L"
    v = fp.commit(str(ds), str(L), mode="copy", quiet=True)

    def evil(rows):
        rows[0]["rest"] = "../../pwned.txt"
        rows[1]["path"] = "../../../pwned2.txt"

    _rewrite_index(L, evil)
    v = fp.load(str(L))
    with pytest.raises(fp.FingerprintError):  # index no longer matches the manifest
        v.checkout(str(tmp_path / "out"))
    assert not list(tmp_path.rglob("pwned*"))


def test_checkout_rejects_traversal_even_with_consistent_manifest(tmp_path, ds):
    """Attacker rewrites index AND manifest checksums consistently: path guard still holds."""
    L = tmp_path / "L"
    fp.commit(str(ds), str(L), mode="copy", quiet=True)
    _rewrite_index(L, lambda rows: rows[0].__setitem__("rest", "../../pwned.txt"))
    v = fp.load(str(L))
    v.manifest["checksums"] = {k: v_ for k, v_ in v.manifest["checksums"].items()}  # untouched digests use path+sha
    with pytest.raises((UnsafePathError, fp.FingerprintError)):
        v.checkout(str(tmp_path / "out"))
    assert not list(tmp_path.rglob("pwned*"))


def test_open_and_verify_refuse_absolute_paths(tmp_path, ds):
    L = tmp_path / "L"
    fp.commit(str(ds), str(L), quiet=True)
    _rewrite_index(L, lambda rows: rows[0].__setitem__("path", "/etc/hostname"))
    v = fp.load(str(L))
    with pytest.raises(UnsafePathError):
        v.open(v.records()[0])
    rep = v.verify()
    assert not rep.ok  # tampered index is reported, file is never read


# ------------------------------------------------------------------ symlinks
@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_outside_source_is_never_read(tmp_path, ds):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    os.symlink(secret, ds / "train" / "leak.txt")
    os.symlink(ds / "train" / "f0.txt", ds / "train" / "inside_link.txt")
    v = fp.commit(str(ds), str(tmp_path / "L"), mode="copy", quiet=True)
    paths = {r["source_path"] for r in v.records()}
    assert "train/leak.txt" not in paths and "train/inside_link.txt" in paths
    assert "TOP SECRET" not in "".join(p.read_text() for p in (tmp_path / "L").rglob("*.txt") if p.is_file())


# ------------------------------------------------------------------ names, tags, versions
@pytest.mark.parametrize("name", ["../evil", "a/b", "", " x", "x;rm -rf"])
def test_invalid_dataset_names(tmp_path, name):
    with pytest.raises(fp.FingerprintError):
        fp.Ledger(str(tmp_path / "L"), name=name or "/")


@pytest.mark.parametrize("tag", ["../prod", "a b", "x/y", "", "1.2.3", "latest"])
def test_invalid_tag_names(tmp_path, ds, tag):
    lg = fp.Ledger(str(tmp_path / "L"))
    lg.commit(str(ds), quiet=True)
    with pytest.raises(fp.FingerprintError):
        lg.tag("latest", tag)


def test_version_string_cannot_inject_paths(tmp_path, ds):
    v = fp.commit(str(ds), str(tmp_path / "L"), version="../../9.9.9", quiet=True)
    assert v.version == "dataset-v9.9.9" and (tmp_path / "L" / "dataset-v9.9.9" / "manifest.json").exists()


# ------------------------------------------------------------------ integrity
def test_load_lock_detects_rewritten_ledger(tmp_path, ds):
    L = tmp_path / "L"
    v = fp.commit(str(ds), str(L), quiet=True)
    lock = v.lock()
    m = L / v.folder / "manifest.json"
    data = json.loads(m.read_text())
    data["fingerprint"] = "sha256:" + "0" * 64
    m.write_text(json.dumps(data))
    with pytest.raises(fp.FingerprintError):
        fp.load_lock(lock)


def test_cache_not_fooled_by_restored_mtime(tmp_path, ds):
    """rsync -t / cp -p / tar restore mtimes; content changes must still be seen."""
    lg = fp.Ledger(str(tmp_path / "L"))
    lg.commit(str(ds), quiet=True)
    f = ds / "test" / "f1.txt"
    st = f.stat()
    f.write_text("test9")  # same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime
    v2 = lg.commit(str(ds), quiet=True)
    assert v2.semver == "1.0.1" and v2["lineage"]["changes"]["modified"] == 1


def test_verify_catches_same_size_same_mtime_edit(tmp_path, ds):
    v = fp.commit(str(ds), str(tmp_path / "L"), quiet=True)
    f = ds / "train" / "f2.txt"
    st = f.stat()
    f.write_text("trainX")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert v.verify().modified == ["train/f2.txt"]


# ------------------------------------------------------------------ secrets & parsers
def test_storage_options_never_written_to_manifest(tmp_path, ds):
    secret = "AKIA-SUPER-SECRET-123"
    v = fp.commit(str(ds), str(tmp_path / "L"), quiet=True, source_storage_options={"key": secret, "secret": secret})
    blob = "".join(p.read_bytes().decode("latin1") for p in (tmp_path / "L").rglob("*") if p.is_file())
    assert secret not in blob and secret not in json.dumps(v.manifest)


def test_npy_probe_refuses_huge_header(tmp_path):
    from dataset_fingerprint.probe import probe_file

    put(tmp_path / "x" / "bad.npy", b"\x93NUMPY\x01\x00" + (65535).to_bytes(2, "little") + b"{" * 100)
    put(tmp_path / "x" / "big.npy", b"\x93NUMPY\x02\x00" + (10**8).to_bytes(4, "little"))
    loc = Location(str(tmp_path / "x"))
    assert "shape" not in probe_file(loc, "bad.npy", "array")
    assert probe_file(loc, "big.npy", "array") == {}


def test_corrupt_inputs_do_not_crash_commit(tmp_path):
    src = tmp_path / "junk"
    put(src / "a.wav", b"RIFF\x00\x00garbage")
    put(src / "b.png", b"\x89PNG not really")
    put(src / "c.npy", b"\x93NUMPY\x01\x00\x05\x00{'x'")
    put(src / "d.csv", "timestamp,x\nnot-a-time,1\nalso-bad,2\n")
    put(src / "e.jsonl", "{not json\n")
    v = fp.commit(str(src), str(tmp_path / "L"), probe=True, label_column="label", quiet=True)
    assert v["metadata"]["total_files"] == 5


# ------------------------------------------------------------------ static guarantees on the source
def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def test_no_dangerous_calls_in_source():
    banned_names = {"eval", "exec", "compile", "__import__"}
    banned_attrs = {("pickle", "load"), ("pickle", "loads"), ("yaml", "load"), ("os", "system"), ("marshal", "loads")}
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for c in _calls(tree):
            if isinstance(c.func, ast.Name):
                assert c.func.id not in banned_names, f"{py.name}: {c.func.id}()"
            if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name):
                assert (c.func.value.id, c.func.attr) not in banned_attrs, f"{py.name}: {c.func.value.id}.{c.func.attr}"
            for kw in c.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False, f"{py.name}: shell=True"


def test_subprocess_only_with_fixed_argv_lists():
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for c in _calls(tree):
            f = c.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "subprocess":
                arg = c.args[0] if c.args else None
                assert isinstance(arg, (ast.List, ast.Name)), f"{py.name}: subprocess without an argv list"


def test_no_network_calls_in_core():
    """The core never phones home: no urllib/requests/http imports outside fsspec drivers."""
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for n in names:
                    assert n.split(".")[0] not in {"requests", "urllib", "http", "socketserver", "httpx", "aiohttp"}, f"{py.name}: {n}"
