from __future__ import annotations

import gzip
import json
import math
import struct
import wave
from pathlib import Path

import pytest

import dataset_fingerprint as fp
from dataset_fingerprint.cli import main as cli
from dataset_fingerprint.splits import detect_split, hash_split, parse_ratios


# ----------------------------------------------------------------- fixtures
def wav(path: Path, seconds=0.5, rate=8000, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(int(rate * seconds))))


def put(path: Path, data: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FINGERPRINT_CACHE", str(tmp_path / "cache.sqlite"))


@pytest.fixture
def image_folder(tmp_path):
    root = tmp_path / "imgs"
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        for cls in ("cat", "dog"):
            for i in range(n):
                put(root / split / cls / f"{cls}_{i}.jpg", f"{split}-{cls}-{i}")
    return root


# ----------------------------------------------------------------- splits
@pytest.mark.parametrize(
    "rel,expected",
    [
        ("train/cat/1.jpg", ("train", "cat/1.jpg")),
        ("audio/validation/x.wav", ("val", "audio/x.wav")),
        ("dev.jsonl", ("val", "dev.jsonl")),
        ("test-00001-of-00004.parquet", ("test", "test-00001-of-00004.parquet")),
        ("raw/clip_1.mp4", (None, "raw/clip_1.mp4")),
        ("trainer_notes/a.txt", (None, "trainer_notes/a.txt")),
    ],
)
def test_detect_split(rel, expected):
    assert detect_split(rel) == expected


def test_parse_ratios():
    assert parse_ratios("0.8,0.1,0.1") == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert parse_ratios("train=7,test=3") == {"train": 0.7, "test": 0.3}
    assert parse_ratios([9, 1]) == {"train": 0.9, "test": 0.1}


def test_hash_split_is_stable_and_balanced():
    r = parse_ratios("0.8,0.1,0.1")
    a = [hash_split(f"f{i}", r) for i in range(5000)]
    assert a == [hash_split(f"f{i}", r) for i in range(5000)]
    assert 0.77 < a.count("train") / 5000 < 0.83


# ----------------------------------------------------------------- commit / load
def test_imagefolder_commit(tmp_path, image_folder):
    v = fp.commit(str(image_folder), str(tmp_path / "ledger"), name="pets", quiet=True)
    assert v.version == "pets-v1.0.0"
    md = v.manifest["metadata"]
    assert md["total_files"] == 20
    assert md["split"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert md["class_distribution"] == {"cat": 0.5, "dog": 0.5}
    assert set(v.manifest["checksums"]) >= {"train_split", "val_split", "test_split", "dataset"}
    loaded = fp.load(str(tmp_path / "ledger"))
    assert loaded.fingerprint == v.fingerprint
    assert len(loaded.files("train")) == 12
    assert all(Path(f).exists() for f in loaded.files())
    assert (tmp_path / "ledger" / "pets-v1.0.0" / "splits" / "train.txt").read_text().count("\n") == 12


def test_no_change_is_idempotent(tmp_path, image_folder):
    lg = fp.Ledger(str(tmp_path / "L"))
    v1 = lg.commit(str(image_folder), quiet=True)
    v2 = lg.commit(str(image_folder), quiet=True)
    assert v1.version == v2.version and len(lg.versions()) == 1


def test_auto_semver_and_lineage(tmp_path, image_folder):
    lg = fp.Ledger(str(tmp_path / "L"), name="ds")
    lg.commit(str(image_folder), quiet=True)
    put(image_folder / "train" / "cat" / "new.jpg", "new")  # new data -> minor
    v = lg.commit(str(image_folder), message="new batch", quiet=True)
    assert v.semver == "1.1.0" and v["lineage"]["changes"]["added"] == 1
    (image_folder / "train" / "dog" / "dog_0.jpg").unlink()  # filtering -> patch
    v = lg.commit(str(image_folder), message="drop corrupt", quiet=True)
    assert v.semver == "1.1.1" and v["lineage"]["previous_version"] == "ds-v1.1.0"
    wav(image_folder / "train" / "cat" / "meow.wav")  # new modality -> major
    v = lg.commit(str(image_folder), quiet=True)
    assert v.semver == "2.0.0"
    d = lg.diff("1.0.0", "latest")
    assert len(d.added) == 2 and len(d.removed) == 1


def test_verify_detects_tampering_reference_mode(tmp_path, image_folder):
    v = fp.commit(str(image_folder), str(tmp_path / "L"), quiet=True)
    assert v.verify().ok
    (image_folder / "val" / "cat" / "cat_0.jpg").write_text("tampered")
    (image_folder / "test" / "dog" / "dog_1.jpg").unlink()
    rep = v.verify()
    assert not rep.ok
    assert rep.modified == ["val/cat/cat_0.jpg"] and rep.missing == ["test/dog/dog_1.jpg"]


def test_copy_mode_snapshot_survives_source_changes(tmp_path, image_folder):
    v = fp.commit(str(image_folder), str(tmp_path / "L"), mode="copy", quiet=True)
    (image_folder / "train" / "cat" / "cat_0.jpg").write_text("changed later")
    assert v.verify().ok
    assert v.files("train")[0].startswith(str(tmp_path / "L" / "dataset-v1.0.0" / "data" / "train"))


def test_index_tamper_detected(tmp_path, image_folder):
    fp.commit(str(image_folder), str(tmp_path / "L"), quiet=True)
    idx = tmp_path / "L" / "dataset-v1.0.0" / "index.jsonl.gz"
    rows = [json.loads(x) for x in gzip.decompress(idx.read_bytes()).decode().splitlines()]
    rows[0]["sha256"] = "0" * 64
    idx.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    rep = fp.load(str(tmp_path / "L")).verify()
    assert rep.index_tampered


def test_unsplit_audio_group_split_no_leakage(tmp_path):
    src = tmp_path / "speech"
    for spk in range(12):
        for utt in range(5):
            wav(src / f"speaker_{spk:02d}" / f"utt_{utt}.wav", seconds=0.1, freq=200 + spk * 30)
    v = fp.commit(str(src), str(tmp_path / "L"), group_by="parent", probe=True, label_from="none", quiet=True)
    assert v["split_strategy"]["type"] == "hash"
    spk_split = {}
    for r in v.records():
        spk_split.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    assert all(len(s) == 1 for s in spk_split.values())  # every speaker in exactly one split
    audio = v["metadata"]["modalities"]["audio"]
    assert audio["files"] == 60 and abs(audio["duration_sec"] - 6.0) < 0.01


def test_new_files_never_reshuffle_existing(tmp_path):
    src = tmp_path / "s"
    for i in range(100):
        put(src / f"doc_{i}.txt", f"text {i}")
    lg = fp.Ledger(str(tmp_path / "L"))
    v1 = {r["source_path"]: r["split"] for r in lg.commit(str(src), quiet=True).records()}
    for i in range(100, 150):
        put(src / f"doc_{i}.txt", f"text {i}")
    v2 = {r["source_path"]: r["split"] for r in lg.commit(str(src), quiet=True).records()}
    assert all(v2[k] == s for k, s in v1.items())


def test_nlp_record_labels(tmp_path):
    src = tmp_path / "nlp"
    put(src / "train.jsonl", "".join(json.dumps({"text": f"t{i}", "label": "pos" if i % 3 else "neg"}) + "\n" for i in range(30)))
    put(src / "test.csv", "text,label\n" + "".join(f"x{i},{'pos' if i % 2 else 'neg'}\n" for i in range(10)))
    v = fp.commit(str(src), str(tmp_path / "L"), label_column="label", probe=True, quiet=True)
    md = v["metadata"]
    assert md["splits"]["train"]["class_counts"] == {"neg": 10, "pos": 20}
    assert md["splits"]["test"]["class_counts"] == {"neg": 5, "pos": 5}
    assert md["label_source"] == "column:label"
    assert md["modalities"]["text"]["records"] == 30


def test_record_model_and_lock(tmp_path, image_folder):
    v = fp.commit(str(image_folder), str(tmp_path / "L"), quiet=True)
    entry = v.record_model("model-v1.0.0", {"acc": 0.94}, run_id="abc")
    assert entry["dataset"]["fingerprint"] == v.fingerprint
    assert fp.load(str(tmp_path / "L")).usage()[0]["metrics"] == {"acc": 0.94}


def test_fsspec_remote_ledger(tmp_path, image_folder):
    pytest.importorskip("fsspec")
    v = fp.commit(str(image_folder), "memory://ledgers/pets", mode="copy", quiet=True)
    assert v.files("train")[0].startswith("memory://")
    assert fp.load("memory://ledgers/pets").verify().ok


def test_databricks_uri_normalisation():
    from dataset_fingerprint.storage import normalize_uri

    assert normalize_uri("dbfs:/mnt/x") in ("/dbfs/mnt/x", "dbfs://mnt/x")
    assert normalize_uri("/Volumes/main/ml/raw") == "/Volumes/main/ml/raw"


def test_cli_roundtrip(tmp_path, image_folder, capsys):
    L = str(tmp_path / "L")
    assert cli(["commit", str(image_folder), "-L", L, "-m", "init", "-t", "resized to 224", "-q"]) == 0
    assert cli(["log", "-L", L]) == 0
    assert cli(["verify", "-L", L]) == 0
    (image_folder / "train" / "cat" / "cat_1.jpg").write_text("x")
    assert cli(["verify", "-L", L]) == 1
    assert cli(["diff", "1.0.0", "-L", L]) == 0
    out = capsys.readouterr().out
    assert "dataset-v1.0.0" in out and "FAILED" in out


# ----------------------------------------------------------------- sensors
def npy(path: Path, shape=(64, 4), dtype="<f4"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode("latin1")
    header += b" " * (64 - (10 + len(header) + 1) % 64) + b"\n"
    n = 1
    for s in shape:
        n *= s
    path.write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + b"\0" * (4 * n))


def imu_csv(path: Path, n=200, hz=100, t0=1_715_000_000_000_000_000):
    rows = ["timestamp_ns,ax,ay,az,gx,gy,gz"]
    rows += [f"{t0 + i * (10**9 // hz)},0.1,0.2,9.8,0,0,0" for i in range(n)]
    put(path, "\n".join(rows) + "\n")


def test_modality_detection_for_sensors():
    from dataset_fingerprint.modality import detect_modality

    assert detect_modality("drive1/radar_front/000001.bin") == "radar"
    assert detect_modality("drive1/imu/accel.csv") == "imu"
    assert detect_modality("drive1/imu0/data.npy") == "imu"
    assert detect_modality("drive1/cam_left/000001.png") == "image"  # camera keeps image
    assert detect_modality("drive1/velodyne/000001.bin") == "lidar"
    assert detect_modality("x/mmwave_scan.rdc", {"radar": "*.rdc"}) == "radar"
    assert detect_modality("labels/train.csv") == "tabular"


def test_multisensor_recordings_stay_together(tmp_path):
    src = tmp_path / "drives"
    for run in range(30):
        base = src / f"2026-05-{run % 28 + 1:02d}_run{run}"
        for f in range(5):
            put(base / "camera_front" / f"{f:06d}.png", f"img{run}{f}")
            npy(base / "radar" / f"{f:06d}.npy", shape=(128, 64))
        imu_csv(base / "imu" / "imu.csv")
    v = fp.commit(str(src), str(tmp_path / "L"), probe=True, quiet=True)
    assert v["split_strategy"]["group_by"] == "auto:recording" and v["split_strategy"]["groups"] == 30
    runs = {}
    for r in v.records():
        runs.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    assert all(len(s) == 1 for s in runs.values())
    md = v["metadata"]
    assert set(md["sensors"]) == {"image", "radar", "imu"}
    assert "num_classes" not in md  # run folders are not classes
    imu = md["modalities"]["imu"]
    assert imu["records"] == 30 * 200 and abs(imu["duration_sec"] - 30 * 1.99) < 0.01
    rec = next(r for r in v.records() if r["modality"] == "imu")
    assert rec["meta"]["sample_rate_hz"] == 100.0
    radar = next(r for r in v.records() if r["modality"] == "radar")
    assert radar["meta"]["shape"] == [128, 64]


def test_single_recording_frames_split_in_synced_blocks(tmp_path):
    src = tmp_path / "rec"
    for f in range(1000):
        put(src / "camera" / f"{f:06d}.jpg", f"c{f}")
        put(src / "radar" / f"{f:06d}.bin", f"r{f}")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    by_frame = {}
    for r in v.records():
        by_frame.setdefault(r["source_path"].split("/")[1].split(".")[0], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_frame.values())  # camera & radar frame N together
    split_of = {int(k): next(iter(s)) for k, s in by_frame.items()}
    gb = v["split_strategy"]["group_by"]
    block = int(gb.split("block=")[1].rstrip(")")) if "block=" in gb else 100
    assert block == 50 and v["split_strategy"]["groups"] == 20
    assert all(len({split_of[i] for i in range(b, b + block)}) == 1 for b in range(0, 1000, block))
    assert v["metadata"]["sequences"]["image"]["frames"] == 1000


def test_frame_sequence_dirs_and_digit_classes(tmp_path):
    src = tmp_path / "seqs"
    for s in range(25):
        for f in range(8):
            put(src / f"seq_{s:03d}" / f"frame_{f:04d}.png", f"{s}-{f}")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    seqsplit = {}
    for r in v.records():
        seqsplit.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    assert all(len(x) == 1 for x in seqsplit.values())
    assert v["metadata"]["sequences"]["image"]["sequences"] == 25
    assert "num_classes" not in v["metadata"]

    mnist = tmp_path / "mnist"
    for d in range(10):
        for i in range(6):
            put(mnist / str(d) / f"{i}.png", f"{d}{i}")
    m = fp.commit(str(mnist), str(tmp_path / "L2"), quiet=True)
    assert m["metadata"]["num_classes"] == 10


def test_few_recordings_still_never_split_apart(tmp_path):
    src = tmp_path / "d"
    for run in range(6):
        for f in range(4):
            put(src / f"run{run}" / "radar" / f"{f:06d}.bin", f"r{run}{f}")
            put(src / f"run{run}" / "camera" / f"{f:06d}.jpg", f"c{run}{f}")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert v["split_strategy"]["group_by"] == "auto:recording"
    runs = {}
    for r in v.records():
        runs.setdefault(r["source_path"].split("/")[0], set()).add(r["split"])
    assert all(len(x) == 1 for x in runs.values())


# ----------------------------------------------------------------- gaps: lineage / augmentation / leakage
def test_augmented_copies_follow_their_original(tmp_path):
    src = tmp_path / "imgs"
    for i in range(60):
        put(src / f"img_{i:03d}.png", f"orig{i}")
        put(src / f"img_{i:03d}_hflip.png", f"flip{i}")
        put(src / "aug" / f"img_{i:03d}_rot90.png", f"rot{i}")
    put(src / "face_mask.png", "not an augmentation")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    by_base = {}
    for r in v.records():
        by_base.setdefault(r.get("aug_of") or r["source_path"].split(".")[0], set()).add(r["split"])
    assert all(len(s) == 1 for k, s in by_base.items() if k.startswith("img"))
    q = v["quality"]
    assert q["augmented_files"] == 120 and q["augmentation_cross_split"] == 0
    assert not any(r.get("augmented") for r in v.records() if "face_mask" in r["source_path"])


def test_augmentation_leak_detected_in_given_splits(tmp_path):
    src = tmp_path / "d"
    for i in range(5):
        put(src / "train" / f"utt{i}.wav", f"a{i}")
        put(src / "test" / f"t{i}.wav", f"t{i}")
    put(src / "train" / "t0_speed1.wav", "leaky")  # augmentation of a TEST file placed in train
    put(src / "test" / "t1_noise.wav", "aug in eval")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert v["quality"]["augmentation_cross_split"] == 1 and v["quality"]["augmented_in_eval"] == 1
    with pytest.raises(fp.FingerprintError):
        fp.commit(str(src), str(tmp_path / "L2"), augmentation="error", quiet=True)


def test_duplicate_content_across_splits(tmp_path):
    src = tmp_path / "d"
    put(src / "train" / "a.jpg", "same")
    put(src / "test" / "b.jpg", "same")
    put(src / "train" / "c.jpg", "other")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert v["quality"]["duplicates_across_splits"] == 1


def test_derived_dataset_parents_trace_and_leak(tmp_path, image_folder):
    raw = fp.commit(str(image_folder), str(tmp_path / "raw"), name="pets", quiet=True)
    aug = tmp_path / "aug_src"
    put(aug / "train" / "cat" / "cat_0_hflip.jpg", "flip of train cat_0")
    put(aug / "train" / "cat" / "copy.jpg", (image_folder / "test" / "cat" / "cat_1.jpg").read_bytes())
    der = fp.commit(
        str(aug), str(tmp_path / "aug"), name="pets-aug", parents=[raw, {"type": "delta", "table": "main.ml.labels", "version": 7}], quiet=True
    )
    assert der["quality"]["parent_eval_leaks"] == 1  # copy of a parent test file
    tree = der.trace()
    assert tree["parents"][0]["dataset_version"] == "pets-v1.0.0"
    assert tree["parents"][1]["table"] == "main.ml.labels"
    assert der.check_lineage() == []
    put(image_folder / "train" / "dog" / "new.jpg", "new")
    fp.commit(str(image_folder), str(tmp_path / "raw"), quiet=True)
    assert der.check_lineage()[0].startswith("stale")
    again = fp.commit(str(aug), str(tmp_path / "aug2"), parents=[f"{tmp_path / 'raw'}@1.0.0"], quiet=True)
    assert again.parents()[0]["dataset_version"] == "pets-v1.0.0"


def test_record_training_multi_dataset_and_load_lock(tmp_path, image_folder):
    a = fp.commit(str(image_folder), str(tmp_path / "A"), name="a", quiet=True)
    src = tmp_path / "b"
    put(src / "train.jsonl", '{"text": "hi"}\n')
    b = fp.commit(str(src), str(tmp_path / "B"), name="b", quiet=True)
    run = fp.record_training("model-v3", [a, b], {"f1": 0.8}, recipe={"hflip": 0.5, "mixup": 0.2}, seed=42)
    assert [d["dataset"] for d in run["datasets"]] == ["a", "b"]
    assert b.usage()[0]["recipe"]["mixup"] == 0.2 and b.usage()[0]["seed"] == 42
    assert fp.load_lock(run["datasets"][0]).fingerprint == a.fingerprint


def test_tags(tmp_path, image_folder):
    lg = fp.Ledger(str(tmp_path / "L"))
    lg.commit(str(image_folder), quiet=True)
    put(image_folder / "train" / "cat" / "z.jpg", "z")
    lg.commit(str(image_folder), quiet=True)
    lg.tag("1.0.0", "prod")
    assert lg.get("prod").semver == "1.0.0"
    lg.tag("latest", "prod")
    assert lg.get("prod").semver == "1.1.0" and lg.tags()["prod"]["history"][0]["version"] == "dataset-v1.0.0"
    with pytest.raises(fp.FingerprintError):
        lg.tag("latest", "2.0.0")


def test_cas_dedups_across_versions_and_checkout(tmp_path, image_folder):
    lg = fp.Ledger(str(tmp_path / "L"))
    v1 = lg.commit(str(image_folder), mode="cas", quiet=True)
    put(image_folder / "train" / "cat" / "extra.jpg", "extra")
    v2 = lg.commit(str(image_folder), mode="cas", quiet=True)
    assert v2["source"]["objects_written"] == 1 and v2["source"]["objects_reused"] == 20
    (image_folder / "train" / "cat" / "cat_0.jpg").write_text("source changed later")
    assert v1.verify().ok
    out = Path(v1.checkout(str(tmp_path / "co"), mode="hardlink"))
    assert (out / "train" / "cat" / "cat_0.jpg").read_text() == "train-cat-0"
    assert (out / "test" / "dog" / "dog_1.jpg").exists()


def test_quick_verify_and_claims(tmp_path, image_folder):
    v = fp.commit(str(image_folder), str(tmp_path / "L"), quiet=True)
    (image_folder / "val" / "dog" / "dog_0.jpg").write_text("different length!")
    assert not v.verify(quick=True).ok
    claims = tmp_path / "L" / ".claims"
    assert not claims.exists() or not any(claims.iterdir())


# ----------------------------------------------------------------- CV / NLP / time series
def test_cv_detection_pairs_stay_together(tmp_path):
    src = tmp_path / "yolo"
    for i in range(80):
        put(src / "images" / f"img_{i:04d}.jpg", f"i{i}")
        put(src / "labels" / f"img_{i:04d}.txt", f"0 0.5 0.5 0.1 0.1 #{i}")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    pairs = {}
    for r in v.records():
        pairs.setdefault(r["source_path"].split("/")[-1].split(".")[0], set()).add(r["split"])
    assert all(len(s) == 1 for s in pairs.values()) and "num_classes" not in v["metadata"]


def test_coco_style_annotation_files(tmp_path):
    src = tmp_path / "coco"
    put(src / "train2017" / "a.jpg", "a")
    put(src / "val2017" / "b.jpg", "b")
    put(src / "annotations" / "instances_train2017.json", "{}")
    put(src / "annotations" / "instances_val2017.json", "{}")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    sp = {r["source_path"]: r["split"] for r in v.records()}
    assert sp["annotations/instances_train2017.json"] == "train" and sp["annotations/instances_val2017.json"] == "val"
    assert sp["train2017/a.jpg"] == "train"


def test_nlp_record_contamination(tmp_path):
    src = tmp_path / "nlp"
    put(src / "train.jsonl", "".join(json.dumps({"text": f"sentence {i}"}) + "\n" for i in range(50)))
    put(src / "test.jsonl", json.dumps({"text": "sentence 7"}) + "\n" + json.dumps({"text": "brand new"}) + "\n")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert v["quality"]["record_overlap_with_train"] == {"test": 1}


def test_temporal_split_for_forecasting(tmp_path):
    src = tmp_path / "sensor_logs"
    for day in range(1, 31):
        imu_csv(src / f"2026-04-{day:02d}" / "imu" / "imu.csv", n=50, hz=10, t0=int((1775001600 + (day - 1) * 86400) * 1e9))
    v = fp.commit(str(src), str(tmp_path / "L"), split="temporal", probe=True, quiet=True)
    days = {r["source_path"][:10]: r["split"] for r in v.records()}
    ordered = [days[k] for k in sorted(days)]
    assert ordered == sorted(ordered, key=["train", "val", "test"].index)  # past -> future
    assert ordered.count("train") == 24 and ordered.count("test") == 3
    tr = v["split_strategy"]["time_ranges"]
    assert tr["train"]["end"] < tr["test"]["start"]
    c = fp.commit(str(src), str(tmp_path / "L2"), time_cutoffs={"val": "2026-04-20", "test": "2026-04-25"}, quiet=True)
    days = {r["source_path"][:10]: r["split"] for r in c.records()}
    assert days["2026-04-19"] == "train" and days["2026-04-20"] == "val" and days["2026-04-25"] == "test"


def test_row_split_single_series(tmp_path):
    src = tmp_path / "one"
    imu_csv(src / "load.csv", n=100, hz=1)
    v = fp.commit(
        str(src),
        str(tmp_path / "L"),
        split="none",
        quiet=True,
        row_split={"column": "timestamp_ns", "val_from": "2026-01-01", "test_from": "2026-03-01"},
    )
    assert v.split_for_time("2025-12-31T23:00:00") == "train"
    assert v.split_for_time("2026-02-01") == "val" and v.split_for_time(1775000000) == "test"


def test_cli_new_commands(tmp_path, image_folder, capsys):
    L = str(tmp_path / "L")
    assert cli(["commit", str(image_folder), "-L", L, "-q", "--mode", "cas"]) == 0
    assert cli(["tag", "prod", "1.0.0", "-L", L]) == 0
    assert cli(["show", "prod", "-L", L]) == 0
    assert cli(["lineage", "-L", L]) == 0
    assert cli(["verify", "prod", "--quick", "-L", L]) == 0
    assert cli(["checkout", "prod", str(tmp_path / "co"), "-L", L]) == 0
    assert (tmp_path / "co" / "train" / "cat" / "cat_0.jpg").exists()


def test_yolo_detection_labels_object_counts(tmp_path):
    src = tmp_path / "det"
    for split, n in (("train2017", 20), ("val2017", 5)):
        for i in range(n):
            put(src / "images" / split / f"{i:012d}.jpg", f"{split}{i}")
            put(src / "labels" / split / f"{i:012d}.txt", "0 0.5 0.5 0.2 0.3\n0 0.1 0.1 0.1 0.1\n2 0.5 0.5 1 1\n")
    v = fp.commit(str(src), str(tmp_path / "L"), label_format="yolo", class_names=["person", "bicycle", "car"], quiet=True)
    md = v["metadata"]
    assert md["splits"]["train"]["class_counts"] == {"car": 20, "person": 40}
    assert md["splits"]["val"]["class_counts"] == {"car": 5, "person": 10}
    assert md["label_source"] == "yolo:objects"
    assert "record_overlap_with_train" not in v["quality"]  # box coordinates are not text records


def test_sparse_numeric_ids_are_not_frame_sequences(tmp_path):
    """COCO / ImageNet style ids (sparse, irregular) are independent samples, not video frames."""
    import random as _r

    rnd = _r.Random(3)
    ids = sorted(rnd.sample(range(1, 600000), 400))
    src = tmp_path / "coco_like"
    for i in ids:
        put(src / "images" / f"{i:012d}.jpg", f"img{i}")
        put(src / "labels" / f"{i:012d}.txt", f"0 0.5 0.5 0.{i % 9 + 1} 0.2\n")
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert "sequences" not in v["metadata"]
    sp = {}
    for r in v.records():
        sp.setdefault(r["source_path"].split("/")[-1].split(".")[0], set()).add(r["split"])
    assert all(len(x) == 1 for x in sp.values())  # image + label still together
    frac = v["metadata"]["split"]
    assert 0.7 < frac["train"] < 0.9  # balanced, not lumped into a few blocks


def test_epoch_ms_frames_are_a_sequence_with_their_step(tmp_path):
    src = tmp_path / "cam"
    t0 = 1_715_000_000_000
    for k in range(600):
        put(src / "front" / f"{t0 + k * 100}.jpg", f"f{k}")  # 10 Hz, ms timestamps
    v = fp.commit(str(src), str(tmp_path / "L"), quiet=True)
    assert v["metadata"]["sequences"]["image"]["frames"] == 600
    by = {}
    for r in v.records():
        by[int(r["source_path"].split("/")[-1].split(".")[0])] = r["split"]
    ks = sorted(by)
    runs = sum(1 for a, b in zip(ks, ks[1:]) if by[a] != by[b])
    assert runs < 40  # contiguous blocks of frames, not frame-by-frame interleaving
