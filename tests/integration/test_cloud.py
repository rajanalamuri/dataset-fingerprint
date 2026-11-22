"""End-to-end behaviour on real object-store APIs (via emulators)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import mlfingerprint as fp

pytestmark = pytest.mark.integration


def seed_drives(c, runs=24, frames=3, prefix="drives"):
    for run in range(runs):
        base = f"{prefix}/2026-04-{run % 28 + 1:02d}_run{run:02d}"
        for f in range(frames):
            c.put(f"{base}/camera/{f:06d}.jpg", f"img-{run}-{f}")
            c.put(f"{base}/radar/{f:06d}.bin", f"radar-{run}-{f}")
        c.put(f"{base}/imu/imu.csv", "timestamp,ax\n" + "".join(f"{1775000000 + run * 86400 + i * 0.01:.2f},0.1\n" for i in range(50)))


# ------------------------------------------------------------------ every backend
def test_reference_commit_verify_tamper_and_patch_bump(cloud):
    seed_drives(cloud)
    L = cloud.uri("ledgers", "drives")
    v = fp.commit(cloud.uri("drives"), L, name="drives", storage_options=cloud.so, probe=True, quiet=True)
    md = v["metadata"]
    assert md["total_files"] == 24 * 7 and set(md["sensors"]) >= {"image", "radar", "imu"}
    assert v["split_strategy"]["group_by"] == "auto:recording"
    runs = {}
    for r in v.records():
        runs.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    assert all(len(s) == 1 for s in runs.values())
    assert fp.load(L, storage_options=cloud.so).verify().ok

    cloud.put("drives/2026-04-01_run00/radar/000000.bin", "TAMPERED!!")
    rep = fp.load(L, storage_options=cloud.so).verify()
    assert not rep.ok and rep.modified == ["2026-04-01_run00/radar/000000.bin"]

    v2 = fp.commit(cloud.uri("drives"), L, storage_options=cloud.so, quiet=True)
    assert v2.semver == "1.0.1" and v2["lineage"]["changes"]["modified"] == 1


def test_cas_snapshot_survives_source_changes_and_checks_out(cloud, tmp_path):
    seed_drives(cloud, runs=6)
    L = cloud.uri("ledgers", "cas")
    v1 = fp.commit(cloud.uri("drives"), L, storage_options=cloud.so, mode="cas", quiet=True)
    cloud.put("drives/2026-04-02_run01/camera/000001.jpg", "overwritten later")
    cloud.put("drives/2026-04-07_run06/camera/000000.jpg", "brand new")
    v2 = fp.commit(cloud.uri("drives"), L, storage_options=cloud.so, mode="cas", quiet=True)
    assert v2.semver == "1.1.0" and v2["source"]["objects_written"] == 2
    assert v1.verify().ok  # old snapshot untouched
    out = Path(v1.checkout(str(tmp_path / "co"), mode="copy"))
    files = [p for p in out.rglob("*") if p.is_file() and p.name != ".mlfingerprint.json"]
    assert len(files) == 6 * 7
    assert any(p.read_text() == "img-1-1" for p in files)  # the pre-overwrite content


def test_temporal_split_tags_and_load_lock(cloud):
    seed_drives(cloud, runs=20, frames=1)
    L = cloud.uri("ledgers", "ts")
    lg = fp.Ledger(L, name="drives", storage_options=cloud.so)
    v = lg.commit(cloud.uri("drives"), split="temporal", probe=True, quiet=True)
    tr = v["split_strategy"]["time_ranges"]
    assert tr["train"]["end"] <= tr["test"]["start"]
    lg.tag("latest", "prod", "integration")
    assert fp.load(L, "prod", storage_options=cloud.so).fingerprint == v.fingerprint
    assert fp.load_lock(v.lock(), storage_options=cloud.so).fingerprint == v.fingerprint


def test_many_objects_parallel_listing(cloud):
    for p in range(20):
        for i in range(60):
            cloud.put(f"big/part{p:02d}/obj{i:03d}.txt", f"{p}-{i}")
    v = fp.commit(cloud.uri("big"), cloud.uri("ledgers", "big"), storage_options=cloud.so, quiet=True, workers=32)
    assert v["metadata"]["total_files"] == 1200


def test_concurrent_commits_never_corrupt(cloud):
    seed_drives(cloud, runs=4, frames=1)
    L = cloud.uri("ledgers", "race")
    results, errors = [], []

    def go():
        try:
            results.append(fp.commit(cloud.uri("drives"), L, storage_options=cloud.so, quiet=True).version)
        except fp.FingerprintError as e:
            errors.append(str(e))
        except Exception as e:  # anything else is a bug: surface it
            errors.append(f"UNEXPECTED {type(e).__name__}: {e}")

    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lg = fp.Ledger(L, storage_options=cloud.so)
    assert len(lg.versions()) == 1 and lg.get().verify().ok
    assert len(results) + len(errors) == 4 and not any(e.startswith("UNEXPECTED") for e in errors), errors


# ------------------------------------------------------------------ cross-cloud + CLI
def test_cross_cloud_s3_source_azure_ledger(s3, azure):
    seed_drives(s3, runs=5, frames=1)
    raw = fp.commit(s3.uri("drives"), s3.uri("ledgers", "raw"), storage_options=s3.so, quiet=True)
    d = fp.commit(
        s3.uri("drives"),
        azure.uri("ledgers", "copy"),
        name="drives-copy",
        source_storage_options=s3.so,
        storage_options=azure.so,
        mode="copy",
        parents=[raw],
        quiet=True,
    )
    back = fp.load(azure.uri("ledgers", "copy"), storage_options=azure.so)
    assert back.verify().ok and back.parents()[0]["fingerprint"] == raw.fingerprint
    assert d.files("train")[0].startswith("az://")


def test_cli_end_to_end_on_s3(s3, s3_server, tmp_path):
    seed_drives(s3, runs=6, frames=1)
    env = dict(os.environ, AWS_ENDPOINT_URL=s3_server, FINGERPRINT_ROOT=s3.uri("ledgers", "cli"))
    run = lambda *a: subprocess.run([sys.executable, "-m", "mlfingerprint", *a], env=env, capture_output=True, text=True)  # noqa: E731
    r = run("commit", s3.uri("drives"), "-n", "drives", "-m", "ci", "--probe", "-q", "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["version"] == "drives-v1.0.0"
    assert run("tag", "prod", "1.0.0").returncode == 0
    assert run("verify", "prod").returncode == 0
    s3.put("drives/2026-04-03_run02/camera/000000.jpg", "edited")
    v = run("verify", "prod")
    assert v.returncode == 1 and "FAILED" in v.stdout
    assert run("checkout", "prod", str(tmp_path / "co"), "--split", "train").returncode == 0
    assert json.loads(run("lock", "prod").stdout)["dataset_version"] == "drives-v1.0.0"
