import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own hash cache - no cross-test leakage, no touching ~/.cache."""
    monkeypatch.setenv("FINGERPRINT_CACHE", str(tmp_path / "hash-cache.sqlite"))
