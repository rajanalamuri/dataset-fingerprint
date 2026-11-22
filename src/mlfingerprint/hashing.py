"""Content hashing, split digests and an mtime-keyed hash cache (so re-committing
a TB-scale dataset only re-hashes files that actually changed)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from typing import IO, Iterable, Optional, Tuple

CHUNK = 4 << 20
ALGORITHM = "sha256"


def hash_stream(f: IO[bytes]) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def digest(items: Iterable[Tuple[str, str]]) -> str:
    """Order-independent digest of (path, file_hash) pairs -> 'sha256:<hex>'.
    Any added, removed, renamed or modified file changes it."""
    h = hashlib.sha256()
    for path, fh in sorted(items):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(fh.encode("ascii"))
        h.update(b"\n")
    return f"{ALGORITHM}:{h.hexdigest()}"


def default_cache_path() -> str:
    env = os.environ.get("FINGERPRINT_CACHE")
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "mlfingerprint", "hashes.sqlite")


class HashCache:
    """sqlite cache: (uri, size, mtime) -> sha256. Safe across threads."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_cache_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS h2 (uri TEXT PRIMARY KEY, size INTEGER, stamp TEXT, sha TEXT)")
        self._pending = 0

    def get(self, uri: str, size: int, stamp: Optional[str]) -> Optional[str]:
        if not stamp:
            return None  # nothing reliable to key on -> always re-hash
        with self._lock:
            row = self._db.execute("SELECT size, stamp, sha FROM h2 WHERE uri=?", (uri,)).fetchone()
        if row and row[0] == size and row[1] == stamp and len(row[2] or "") == 64:
            return row[2]
        return None

    def put(self, uri: str, size: int, stamp: Optional[str], sha: str) -> None:
        if not stamp:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO h2 VALUES (?,?,?,?)", (uri, size, stamp, sha))
            self._pending += 1
            if self._pending >= 500:
                self._db.commit()
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()
