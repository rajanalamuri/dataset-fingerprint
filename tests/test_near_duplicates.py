import io
import random

import pytest

from mlfingerprint import FingerprintError, Ledger, perceptual

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageEnhance  # noqa: E402


def _img(seed, size=64):
    rnd = random.Random(seed)
    im = Image.new("RGB", (8, 8))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(64)])
    return im.resize((size, size), Image.BICUBIC)


def test_pigeonhole_search_matches_brute_force():
    rnd = random.Random(1)
    base = [rnd.getrandbits(64) for _ in range(300)]
    items = []
    for i, h in enumerate(base):
        items.append(("train", h))
        if i % 3 == 0:  # a noisy copy in test
            for b in rnd.sample(range(64), rnd.randrange(0, 7)):
                h ^= 1 << b
            items.append(("test", h))
    for t in (0, 3, 5):
        got, _ = perceptual.cross_split_pairs(items, t)
        want = {
            (i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if items[i][0] != items[j][0] and bin(items[i][1] ^ items[j][1]).count("1") <= t
        }
        assert {(i, j) for i, j, _ in got} == want


def test_dhash_is_stable_under_reencoding():
    im = _img(3)
    raw, jpg = io.BytesIO(), io.BytesIO()
    im.save(raw, "PNG")
    ImageEnhance.Brightness(im).enhance(1.1).resize((48, 48)).save(jpg, "JPEG", quality=70)
    a, b = perceptual.dhash(io.BytesIO(raw.getvalue())), perceptual.dhash(io.BytesIO(jpg.getvalue()))
    assert a is not None and bin(a ^ b).count("1") <= 4
    assert perceptual.dhash(io.BytesIO(b"not an image")) is None


def test_commit_flags_near_duplicate_across_splits(tmp_path):
    src = tmp_path / "data"
    for sp in ("train", "test"):
        (src / sp / "cat").mkdir(parents=True)
    for i in range(20):
        _img(i).save(src / "train" / "cat" / f"{i}.png")
    for i in range(100, 110):
        _img(i).save(src / "test" / "cat" / f"{i}.png")
    copy = _img(7)  # train image 7, brightened + resized + JPEG -> different bytes
    ImageEnhance.Brightness(copy).enhance(1.08).resize((50, 50)).save(src / "test" / "cat" / "leak.jpg", quality=75)

    lg = Ledger(str(tmp_path / "ledger"))
    v = lg.commit(str(src), near_duplicates=True, quiet=True, cache=False)
    nd = v.manifest["quality"]["near_duplicates"]
    assert nd["with_train_match"] == {"test": 1}
    assert {nd["examples"][0][0], nd["examples"][0][1]} == {"train/cat/7.png", "test/cat/leak.jpg"}
    assert v.manifest["quality"]["duplicates_across_splits"] == 0
    assert any("near-duplicate" in w for w in v.manifest["lineage"]["warnings"])

    v0 = Ledger(str(tmp_path / "l2")).commit(str(src), quiet=True, cache=False)  # off by default
    assert "near_duplicates" not in v0.manifest["quality"]


def test_near_duplicates_without_pillow(tmp_path, monkeypatch):
    src = tmp_path / "d" / "train"
    src.mkdir(parents=True)
    _img(1).save(src / "a.png")
    monkeypatch.setattr(perceptual, "available", lambda: False)
    v = Ledger(str(tmp_path / "l")).commit(str(tmp_path / "d"), near_duplicates=True, quiet=True, cache=False)
    assert "near_duplicates" not in v.manifest["quality"]


def test_threshold_validation(tmp_path):
    with pytest.raises(FingerprintError):
        Ledger(str(tmp_path / "l")).commit(str(tmp_path), near_duplicates=99, quiet=True)
