"""End-to-end POC: one multimodal dataset (video + audio + images + NLP captions,
later + lidar) evolving over 4 versions, an unsplit speech corpus, and multi-sensor
drive recordings (camera frames at 10 fps + radar + 200 Hz IMU + microphone).

    python examples/poc_multimodal.py /tmp/dataset-fingerprint-poc

Needs only the standard library; uses ffmpeg for real MP4s and Pillow for real
PNGs when available (falls back to placeholder bytes otherwise).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import struct
import subprocess
import sys
import wave
import zlib
from pathlib import Path

import dataset_fingerprint as fp

CLASSES = ["car", "pedestrian", "cyclist"]
rng = random.Random(7)


# ---------------------------------------------------------------- synthetic data
def png(path: Path, w=64, h=48, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = random.Random(seed)
    raw = b"".join(b"\0" + bytes(r.randrange(256) for _ in range(w * 3)) for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def wav(path: Path, seconds=1.0, freq=440, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(int(rate * seconds))))


def mp4(path: Path, seconds=2, size="160x120", seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={seconds}:size={size}:rate=10",
                "-vf",
                f"hue=h={seed * 40}",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
    else:
        path.write_bytes(f"fake-mp4-{seed}".encode())


def npy_write(path: Path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = repr({"descr": "<f4", "fortran_order": False, "shape": tuple(shape)}).encode("latin1")
    header += b" " * (64 - (10 + len(header) + 1) % 64) + b"\n"
    n = 1
    for d in shape:
        n *= d
    path.write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + bytes(4 * n))


def captions(path: Path, n: int, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for i in range(n):
            c = CLASSES[(i + offset) % 3]
            f.write(json.dumps({"text": f"a {c} crossing the junction #{i + offset}", "label": c}) + "\n")


def make_batch(root: Path, batch: int, n_img=8, n_aud=3, n_vid=1):
    for split, frac in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
        for ci, c in enumerate(CLASSES):
            for i in range(max(1, int(n_img * frac))):
                png(root / split / "images" / c / f"b{batch}_{i}.png", seed=batch * 1000 + ci * 100 + i)
            for i in range(max(1, int(n_aud * frac))):
                wav(root / split / "audio" / c / f"b{batch}_{i}.wav", seconds=0.5 + i * 0.25, freq=300 + 100 * ci)
            for i in range(n_vid):
                mp4(root / split / "video" / c / f"b{batch}_{i}.mp4", seconds=2, seed=ci + batch)
        captions(root / split / "captions.jsonl", 20 if split == "train" else 5, offset=batch * 7)


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ---------------------------------------------------------------- the POC
def main(work: Path):
    if work.exists():
        shutil.rmtree(work)
    data, ledger_root = work / "av_dataset", work / "ledger"
    lg = fp.Ledger(str(ledger_root), name="av-multimodal")
    opts = dict(probe=True, label_column="label", quiet=True)

    banner("1. Initial release - video + audio + images + NLP captions, train/val/test dirs")
    make_batch(data, batch=1)
    v = lg.commit(
        str(data),
        message="initial release",
        transformations=["resized frames to 64x48", "audio -> 16kHz mono"],
        source_batches=["raw-data/batch-2026-05-08/"],
        **opts,
    )
    print(v.summary())

    banner("2. New collection batch arrives -> auto MINOR bump")
    make_batch(data, batch=2)
    v = lg.commit(str(data), message="added batch-2026-05-09", source_batches=["raw-data/batch-2026-05-09/"], **opts)
    print(v.summary())

    banner("3. Filter corrupt files -> auto PATCH bump")
    for p in sorted((data / "train" / "images" / "car").glob("b2_*.png"))[:2]:
        p.unlink()
    v = lg.commit(
        str(data),
        message="removed 2 corrupt frames",
        transformations=[{"name": "filter_corrupt_frames", "removed_frames": 2, "reason": "CRC mismatch"}],
        **opts,
    )
    print(v.summary())

    banner("4. Same data again -> nothing to commit (idempotent, CI-safe)")
    same = lg.commit(str(data), **opts)
    print(f"returned existing {same.version}; versions on disk: {lg.versions()}")

    banner("5. New sensor modality (lidar .pcd) -> auto MAJOR bump")
    for split in ("train", "val", "test"):
        for i in range(3):
            p = data / split / "lidar" / f"sweep_{i}.pcd"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# .PCD v0.7\nPOINTS {100 + i}\n")
    v = lg.commit(str(data), message="added lidar sweeps (new collection rig)", **opts)
    print(v.summary())

    banner("6. Why is v2.0.0 different from v1.0.0?  -> diff")
    print(lg.diff("1.0.0", "2.0.0").render(limit=4))

    banner("7. Training job: load by version, use files, record the model")
    ds = fp.load(str(ledger_root), "1.1.1")
    train = list(ds.samples("train"))
    print(f"{ds.version}: {len(train)} train samples, e.g. {json.dumps(train[0])[:150]}...")
    entry = ds.record_model("model-v1.0.0", metrics={"accuracy": 0.94}, run_id="mlflow:42")
    print("model metadata to store with the model:\n" + json.dumps(entry["dataset"], indent=2))

    banner("8. Someone edits a training file in place -> verify catches it")
    victim = data / "val" / "audio" / "pedestrian" / "b1_0.wav"
    victim.write_bytes(victim.read_bytes()[:-100] + b"\0" * 100)
    print(lg.verify("1.1.1").render())

    banner("9. Unsplit speech corpus: speaker-grouped stable hash split (no speaker leakage)")
    speech = work / "speech_raw"
    for s in range(10):
        for u in range(4):
            wav(speech / f"spk{s:02d}" / f"utt{u}.wav", seconds=0.3, freq=150 + 20 * s)
    sv = fp.commit(
        str(speech), str(work / "speech_ledger"), name="speech", group_by="parent", split="0.8,0.1,0.1", probe=True, label_from="none", quiet=True
    )
    print(sv.summary())
    groups = {}
    for r in sv.records():
        groups.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    print("speakers in >1 split:", [k for k, s in groups.items() if len(s) > 1] or "none")

    banner("10. Multi-sensor drives: camera frames @10fps + radar + IMU + mic, no splits given")
    drives = work / "drives_raw"
    for run in range(24):
        base = drives / f"2026-05-{run % 28 + 1:02d}_run{run:02d}"
        clip = work / "tmp" / f"run{run}.mp4"
        mp4(clip, seconds=1, seed=run)
        (base / "camera_front").mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):  # video divided into frames at a definite fps
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", "fps=10", str(base / "camera_front" / "%06d.jpg")], check=True)
        n_frames = len(list((base / "camera_front").glob("*.jpg"))) or 10
        for f in range(1, n_frames + 1):
            npy_write(base / "radar_front" / f"{f:06d}.npy", (64, 128))
        t0 = 1_715_000_000_000_000_000 + run * 10**12
        rows = ["timestamp_ns,ax,ay,az,gx,gy,gz"] + [
            f"{t0 + i * 5_000_000},{rng.gauss(0, 0.1):.4f},{rng.gauss(0, 0.1):.4f},{9.81 + rng.gauss(0, 0.05):.4f},0,0,0" for i in range(200)
        ]  # 200 Hz accelerometer / gyro
        (base / "imu").mkdir(parents=True, exist_ok=True)
        (base / "imu" / "imu.csv").write_text("\n".join(rows) + "\n")
        wav(base / "mic" / "cabin.wav", seconds=1.0, freq=120)
    shutil.rmtree(work / "tmp", ignore_errors=True)
    drv = fp.commit(
        str(drives),
        str(work / "drives_ledger"),
        name="drives",
        probe=True,
        quiet=True,
        message="24 drive recordings",
        extra={"camera_fps": 10, "imu_rate_hz": 200},
    )
    print(drv.summary())
    print("split strategy:", json.dumps(drv["split_strategy"]))
    print("sequences     :", json.dumps(drv["metadata"]["sequences"]))
    imu = next(r for r in drv.records() if r["modality"] == "imu")
    radar = next(r for r in drv.records() if r["modality"] == "radar")
    print("imu file meta :", json.dumps(imu["meta"]))
    print("radar meta    :", json.dumps(radar["meta"]))
    runs = {}
    for r in drv.records():
        runs.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    print("recordings split across >1 split:", [k for k, x in runs.items() if len(x) > 1] or "none")

    banner("11. Offline augmentation -> derived dataset with parents, leak checks, lineage")
    aug = work / "av_aug"
    raw_v = fp.load(str(ledger_root), "2.0.0")
    for r in raw_v.records():
        if r["modality"] == "image" and r["split"] in ("train", "test"):
            src_p = data / r["source_path"]
            if src_p.exists():
                dst = aug / "train" / "images" / r["label"] / (src_p.stem + "_hflip.png")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src_p.read_bytes()[::-1])  # stand-in for a flipped image
    der = fp.commit(str(aug), str(work / "aug_ledger"), name="av-aug", parents=[raw_v], transformations=[{"name": "hflip", "p": 1.0}], quiet=True)
    print(der.summary())
    print(json.dumps(der.trace(), indent=2)[:400])

    banner("12. One training run on several datasets + on-the-fly augmentation recipe")
    run = fp.record_training(
        "model-v2.0.0", [raw_v, der], metrics={"mAP": 0.61}, recipe={"random_crop": 0.5, "color_jitter": 0.2, "specaugment": True}, seed=1234
    )
    print(json.dumps({k: run[k] for k in ("model_version", "recipe", "seed")}))
    print("datasets:", [d["dataset_version"] for d in run["datasets"]])
    print("reopen exact data from the stored lock:", fp.load_lock(run["datasets"][0]))

    banner("13. Time-series forecasting: temporal split (past -> train, future -> test)")
    ts = work / "energy_load"
    for day in range(1, 31):
        rows = ["timestamp,load_kw"] + [
            f"2026-04-{day:02d}T{h:02d}:00:00,{50 + 10 * math.sin(h / 24 * 6.28) + rng.gauss(0, 2):.2f}" for h in range(24)
        ]
        (ts / "site_A" / f"2026-04-{day:02d}.csv").parent.mkdir(parents=True, exist_ok=True)
        (ts / "site_A" / f"2026-04-{day:02d}.csv").write_text("\n".join(rows) + "\n")
    tv = fp.commit(str(ts), str(work / "ts_ledger"), name="energy", split="temporal:0.7,0.15,0.15", probe=True, quiet=True)
    print(tv.summary())
    print("time ranges:", json.dumps(tv["split_strategy"]["time_ranges"]))

    banner("14. NLP: test contamination (same sentence in train and test)")
    nlp = work / "nlp"
    nlp.mkdir(parents=True, exist_ok=True)
    (nlp / "train.jsonl").write_text(
        "".join(json.dumps({"text": f"the {c} turned left at junction {i}", "label": c}) + "\n" for i, c in enumerate(CLASSES * 30))
    )
    (nlp / "test.jsonl").write_text(
        json.dumps({"text": "The car   turned left at junction 0", "label": "car"})
        + "\n"
        + json.dumps({"text": "a cyclist waits", "label": "cyclist"})
        + "\n"
    )
    nv = fp.commit(str(nlp), str(work / "nlp_ledger"), name="nlp", label_column="label", probe=True, quiet=True)
    print("quality:", json.dumps({k: v for k, v in nv["quality"].items() if v and k != "examples"}))
    print("warnings:", nv["lineage"]["warnings"])

    banner("15. Tags + content-addressed snapshots + checkout")
    lg.tag("1.1.1", "prod", "model-v1.0.0 shipped on this")
    print("prod ->", lg.get("prod").version)
    cas = fp.Ledger(str(work / "cas_ledger"), name="av")
    c1 = cas.commit(str(data), mode="cas", quiet=True)
    (data / "train" / "images" / "car" / "extra.png").write_bytes(b"new frame")
    c2 = cas.commit(str(data), mode="cas", quiet=True)
    print(f"{c2.version}: objects written {c2['source']['objects_written']}, reused {c2['source']['objects_reused']} (dedup)")
    print("checkout ->", c1.checkout(str(work / "checkout_v1"), split="val", mode="hardlink"))

    banner("16. Ledger history")
    from dataset_fingerprint.cli import main as cli

    cli(["log", "-L", str(ledger_root)])
    print(f"\nmanifest: {ledger_root / 'av-multimodal-v2.0.0' / 'manifest.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dataset-fingerprint-poc"))
