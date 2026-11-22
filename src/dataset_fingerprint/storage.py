"""Minimal filesystem abstraction: local paths out of the box, any fsspec URI
(s3://, gs://, az://, hf://, ...) when fsspec is installed."""

from __future__ import annotations

import fnmatch
import io
import os
import posixpath
import re
import shutil
from dataclasses import dataclass
from typing import IO, Iterator, Optional, Sequence

IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", "__pycache__", ".git", ".ipynb_checkpoints"}


def is_remote(uri: str) -> bool:
    uri = normalize_uri(uri)
    return "://" in uri and not uri.startswith("file://")


def normalize_uri(uri: str) -> str:
    """Make Databricks / Azure spellings uniform.

    - ``dbfs:/path`` on a Databricks cluster -> the ``/dbfs/path`` FUSE mount (fast, local);
      off-cluster -> ``dbfs://path`` handled by fsspec's Databricks filesystem
      (needs ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` or storage_options instance/token).
    - Unity Catalog volumes (``/Volumes/cat/schema/vol/...``) are plain local paths on clusters.
    - ``abfss://`` / ``abfs://`` / ``az://`` go to adlfs.
    """
    uri = str(uri)
    if uri.startswith("dbfs:/") and not uri.startswith("dbfs://"):
        if os.path.isdir("/dbfs"):
            return "/dbfs/" + uri[len("dbfs:/") :].lstrip("/")
        return "dbfs://" + uri[len("dbfs:/") :].lstrip("/")
    return uri


def _remote_options(protocol: str, opts: dict) -> dict:
    """Fill in sensible defaults from the environment so URIs 'just work'."""
    opts = dict(opts or {})
    if protocol == "dbfs":
        opts.setdefault("instance", os.environ.get("DATABRICKS_HOST", "").replace("https://", "").rstrip("/"))
        opts.setdefault("token", os.environ.get("DATABRICKS_TOKEN"))
    if protocol in ("gs", "gcs") and os.environ.get("STORAGE_EMULATOR_HOST"):
        # gcsfs >= 2025 probes bucket type over gRPC first; emulators don't speak it and every call
        # stalls ~60 s before falling back. Must be set before gcsfs is imported.
        os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")
    if protocol == "s3" and os.environ.get("AWS_ENDPOINT_URL") and "endpoint_url" not in opts:
        # MinIO, Cloudflare R2, Ceph, LocalStack, ...
        opts.setdefault("client_kwargs", {}).setdefault("endpoint_url", os.environ["AWS_ENDPOINT_URL"])
    return opts


_EXTRA_FOR_PROTOCOL = {
    "s3": "s3",
    "s3a": "s3",
    "gs": "gcs",
    "gcs": "gcs",
    "az": "azure",
    "abfs": "azure",
    "abfss": "azure",
    "adl": "azure",
    "dbfs": "databricks",
    "hf": "hf",
}


class UnsafePathError(ValueError):
    """A path from a manifest/index tried to escape its root (absolute, '..', drive letter, NUL)."""


_DRIVE = re.compile(r"^[A-Za-z]:")


def safe_rel(rel: str) -> str:
    """Validate a relative posix path coming from (possibly tampered) metadata."""
    if not isinstance(rel, str) or "\x00" in rel or "\\" in rel or rel.startswith("/") or _DRIVE.match(rel):
        raise UnsafePathError(f"unsafe path: {rel!r}")
    if any(part in ("..",) for part in rel.split("/")):
        raise UnsafePathError(f"path escapes its root: {rel!r}")
    return rel


@dataclass
class FileEntry:
    rel: str  # posix path relative to the location root
    size: int
    mtime: Optional[float]
    stamp: Optional[str] = None  # change detector for the hash cache (mtime+ctime+inode / ETag)


class Location:
    """A root directory, local or remote. All `rel` paths are posix-style."""

    def __init__(self, uri: str, storage_options: Optional[dict] = None):
        uri = normalize_uri(str(uri))
        if uri.startswith("file://"):
            uri = uri[len("file://") :]
        self.storage_options = dict(storage_options or {})
        self.remote = is_remote(uri)
        if self.remote:
            self.protocol = uri.split("://", 1)[0]
            extra = _EXTRA_FOR_PROTOCOL.get(self.protocol, "cloud")
            try:
                import fsspec

                self.fs, path = fsspec.core.url_to_fs(uri, **_remote_options(self.protocol, self.storage_options))
            except ImportError as e:  # pragma: no cover - depends on env
                raise ImportError(f"'{uri}' needs an extra driver: pip install 'dataset-fingerprint[{extra}]'  ({e})") from e
            self.root = path.rstrip("/")
            self.refresh()
        else:
            self.protocol = "file"
            self.fs = None
            self.root = os.path.abspath(os.path.expanduser(uri))

    def refresh(self) -> None:
        """Drop fsspec's listing cache - remote stores change under us (new batches, tampering)."""
        if self.remote:
            try:
                self.fs.invalidate_cache()
            except Exception:  # some fsspec backends have no listing cache  # nosec B110
                pass

    # ---------------------------------------------------------------- paths
    def _p(self, rel: str = "") -> str:
        if not rel:
            return self.root
        safe_rel(rel)
        if self.remote:
            return posixpath.join(self.root, rel)
        return os.path.join(self.root, *rel.split("/"))

    def uri(self, rel: str = "") -> str:
        p = self._p(rel)
        return f"{self.protocol}://{p}" if self.remote else p

    def child(self, rel: str) -> "Location":
        return Location(self.uri(rel), self.storage_options)

    @property
    def is_local(self) -> bool:
        return not self.remote

    def local_path(self, rel: str = "") -> Optional[str]:
        return None if self.remote else self._p(rel)

    # ------------------------------------------------------------------- io
    def exists(self, rel: str = "") -> bool:
        return self.fs.exists(self._p(rel)) if self.remote else os.path.exists(self._p(rel))

    def isdir(self, rel: str = "") -> bool:
        return self.fs.isdir(self._p(rel)) if self.remote else os.path.isdir(self._p(rel))

    def open(self, rel: str, mode: str = "rb") -> IO:
        if self.remote:
            return self.fs.open(self._p(rel), mode)
        if "w" in mode or "a" in mode:
            os.makedirs(os.path.dirname(self._p(rel)), exist_ok=True)
        return open(self._p(rel), mode)

    def size(self, rel: str) -> int:
        return int(self.fs.size(self._p(rel))) if self.remote else os.path.getsize(self._p(rel))

    def read_text(self, rel: str) -> str:
        with self.open(rel, "rb") as f:
            return f.read().decode("utf-8")

    def write_text(self, rel: str, text: str) -> None:
        self.write_bytes(rel, text.encode("utf-8"))

    def write_bytes(self, rel: str, data: bytes) -> None:
        if self.remote:
            with self.fs.open(self._p(rel), "wb") as f:
                f.write(data)
            return
        path = self._p(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)  # atomic on local disks

    def listdir(self, rel: str = "") -> list:
        if not self.exists(rel):
            return []
        if self.remote:
            return sorted(posixpath.basename(p.rstrip("/")) for p in self.fs.ls(self._p(rel), detail=False))
        return sorted(os.listdir(self._p(rel)))

    def _parallel_find(self, workers: int = 32) -> dict:
        """List a bucket prefix by listing each top-level sub-prefix concurrently -
        big win on object stores with millions of keys."""
        from concurrent.futures import ThreadPoolExecutor

        try:
            top = self.fs.ls(self.root, detail=True)
        except FileNotFoundError:
            return {}
        found = {e["name"]: e for e in top if e.get("type") != "directory"}
        dirs = [e["name"] for e in top if e.get("type") == "directory"]
        if not dirs:
            return found
        with ThreadPoolExecutor(max_workers=min(workers, len(dirs))) as ex:
            for part in ex.map(lambda d: self.fs.find(d, detail=True), dirs):
                found.update(part)
        return found

    def walk(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        skip_dirs: Sequence[str] = (),
    ) -> Iterator[FileEntry]:
        """Yield every file under the root (hidden files and OS junk skipped)."""
        skip_abs = {os.path.abspath(d) for d in skip_dirs if not is_remote(d)}

        def keep(rel: str) -> bool:
            parts = rel.split("/")
            if any(p.startswith(".") or p in IGNORED_NAMES for p in parts):
                return False
            if include and not any(fnmatch.fnmatch(rel, pat) for pat in include):
                return False
            if exclude and any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                return False
            return True

        if self.remote:
            self.refresh()
            found = self._parallel_find()
            for path, info in sorted(found.items()):
                if info.get("type") == "directory":
                    continue
                rel = path[len(self.root) :].lstrip("/")
                if keep(rel):
                    mtime = info.get("ETag") or info.get("etag") or info.get("mtime") or info.get("LastModified") or info.get("last_modified")
                    stamp = info.get("ETag") or info.get("etag") or info.get("md5") or info.get("content_md5") or mtime
                    yield FileEntry(rel, int(info.get("size") or 0), _as_float(mtime), None if stamp is None else str(stamp))
            return

        self.skipped_symlinks = []
        real_root = os.path.realpath(self.root)
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in IGNORED_NAMES and os.path.abspath(os.path.join(dirpath, d)) not in skip_abs
            )
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                if not keep(rel):
                    continue
                if os.path.islink(full):
                    target = os.path.realpath(full)
                    if os.path.commonpath([real_root, target]) != real_root or not os.path.isfile(target):
                        self.skipped_symlinks.append(rel)  # never read files outside the dataset root
                        continue
                st = os.stat(full)
                # ctime + inode catch content changes even when a tool restores mtime (rsync -t, cp -p, tar)
                stamp = f"{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}"
                if os.name == "nt":  # Windows has no change-time/inode: add a head+tail content sample
                    stamp += ":" + _sample_digest(full, st.st_size)
                yield FileEntry(rel, st.st_size, st.st_mtime, stamp)


def _sample_digest(path: str, size: int, chunk: int = 64 << 10) -> str:
    """First + last 64 KB (whole file when small): catches restored-mtime edits on Windows cheaply."""
    import hashlib

    h = hashlib.blake2b(digest_size=12)
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
        elif size > chunk:
            h.update(f.read())
    return h.hexdigest()


def _as_float(v) -> Optional[float]:  # noqa: E302
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "timestamp"):
        return v.timestamp()
    # ETag / string: hash it to a stable number so it still invalidates the cache
    import hashlib

    return float(int(hashlib.sha256(str(v).encode()).hexdigest()[:13], 16))


def copy_file(src: Location, src_rel: str, dst: Location, dst_rel: str, mode: str = "copy") -> None:
    """Copy (or hardlink) a file between two locations."""
    if src.is_local and dst.is_local:
        s, d = src._p(src_rel), dst._p(dst_rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(s, d)
                return
            except OSError:
                pass  # cross-device etc. -> fall back to copy
        shutil.copy2(s, d)
        return
    if src.remote and dst.remote and src.fs is dst.fs:
        src.fs.copy(src._p(src_rel), dst._p(dst_rel))
        return
    with src.open(src_rel, "rb") as fi, dst.open(dst_rel, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=8 << 20)


def read_bytes(loc: Location, rel: str) -> bytes:
    with loc.open(rel, "rb") as f:
        return f.read()


__all__ = ["Location", "FileEntry", "copy_file", "is_remote", "read_bytes", "io"]
