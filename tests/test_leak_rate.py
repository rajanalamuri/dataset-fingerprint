import pytest

from dataset_fingerprint import FingerprintError, Ledger
from dataset_fingerprint.cli import _rate


def _w(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_leak_rate_counts_exact_and_augmented_copies(tmp_path):
    src = tmp_path / "d"
    for i in range(8):
        _w(src / "train" / "cat" / f"img_{i}.png", b"train%d" % i)
    for i in range(10, 20):
        _w(src / "test" / "cat" / f"img_{i}.png", b"test%d" % i)
    _w(src / "test" / "cat" / "copy.png", b"train3")  # exact copy of a train file
    _w(src / "train" / "cat" / "img_12_hflip.png", b"flipped12")  # augmentation of a test file
    v = Ledger(str(tmp_path / "l")).commit(str(src), quiet=True, cache=False)
    lk = v.manifest["quality"]["leakage"]
    t = lk["by_split"]["test"]
    assert t["files"] == 11 and t["leaked"] == 2 and t["effective_size"] == 9
    assert t["by_type"] == {"exact_copy": 1, "augmented_copy": 1}
    assert lk["leak_rate"] == round(2 / 11, 6)
    assert "leak rate" in v.summary()


def test_clean_dataset_has_zero_rate_and_gate(tmp_path):
    src = tmp_path / "d"
    for i in range(5):
        _w(src / "train" / f"{i}.wav", b"a%d" % i)
        _w(src / "val" / f"{i}.wav", b"b%d" % i)
    v = Ledger(str(tmp_path / "l")).commit(str(src), quiet=True, cache=False, max_leak_rate=0.0)
    assert v.manifest["quality"]["leakage"]["leak_rate"] == 0.0
    _w(src / "val" / "dup.wav", b"a1")
    with pytest.raises(FingerprintError, match="exceeds max_leak_rate"):
        Ledger(str(tmp_path / "l")).commit(str(src), quiet=True, cache=False, max_leak_rate=0.05)


def test_record_leak_rate_for_text(tmp_path):
    src = tmp_path / "d"
    _w(src / "train" / "a.jsonl", b'{"t": "hello"}\n{"t": "world"}\n')
    _w(src / "test" / "b.jsonl", b'{"t": "hello"}\n{"t": "new"}\n{"t": "other"}\n{"t": "x"}\n')
    t = Ledger(str(tmp_path / "l")).commit(str(src), quiet=True, cache=False).manifest["quality"]["leakage"]["by_split"]["test"]
    assert t["records"] == 4 and t["records_also_in_train"] == 1 and t["record_leak_rate"] == 0.25


def test_no_train_split_means_no_rate(tmp_path):
    _w(tmp_path / "d" / "test" / "a.png")
    v = Ledger(str(tmp_path / "l")).commit(str(tmp_path / "d"), quiet=True, cache=False)
    assert v.manifest["quality"]["leakage"]["leak_rate"] is None


def test_rate_parser():
    assert _rate("0.5%") == 0.005 and _rate("0.01") == 0.01 and _rate(None) is None


def test_same_recording_name_in_train_and_test_is_a_warning_not_a_leak(tmp_path):
    src = tmp_path / "d"
    for sp, seqs in (("train", ("seq01", "seq02")), ("test", ("seq01", "seq09"))):
        for s in seqs:
            for f in range(1, 11):
                _w(src / sp / s / f"{f:06d}.png", f"{sp}{s}{f}".encode())
    v = Ledger(str(tmp_path / "l")).commit(str(src), quiet=True, cache=False, label_from="none")
    lk = v.manifest["quality"]["leakage"]
    assert lk["leak_rate"] == 0.0
    assert lk["recording_names_also_in_train"]["count"] == 1
    assert any("seq01" in w for w in v.manifest["lineage"]["warnings"])


def test_takes_of_one_source_clip_stay_together_and_are_flagged_when_split(tmp_path):
    from dataset_fingerprint.splits import source_stem

    assert source_stem("1-100038-A-14") == "1-100038-14" and source_stem("rec7_part2") == "rec7"
    assert source_stem("frame_000123") == "frame_000123" and source_stem("a") == "a"
    flat = tmp_path / "audio"
    for c in range(40):
        for t in "ABC":
            _w(flat / f"1-{1000 + c}-{t}-7.wav", f"{c}{t}".encode())
    v = Ledger(str(tmp_path / "l")).commit(str(flat), split="0.7,0.3", quiet=True, cache=False)
    by_clip = {}
    for r in v.records():
        by_clip.setdefault(r["source_path"].split("-")[1], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_clip.values())

    given = tmp_path / "given"  # a file-level split handed over as folders: takes end up on both sides
    for c in range(10):
        _w(given / "train" / f"1-{c}-A-7.wav", b"a%d" % c)
        _w(given / "test" / f"1-{c}-B-7.wav", b"b%d" % c)
    v = Ledger(str(tmp_path / "l2")).commit(str(given), quiet=True, cache=False, label_from="none")
    assert v.manifest["quality"]["leakage"]["recording_names_also_in_train"]["count"] == 10
