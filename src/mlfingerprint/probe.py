"""Optional, dependency-light per-file metadata: audio/video duration, image size,
text record counts and record-level label counts. Every probe fails soft."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess  # only ffprobe with a fixed argv  # nosec B404
import wave
from collections import Counter
from typing import Any, Dict, Optional

from .modality import extension
from .storage import Location

_FFPROBE = shutil.which("ffprobe")


_IMAGE = {"jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp", "ppm", "pgm"}
_AUDIO = {"wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "opus", "aiff", "aif"}
_VIDEO = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg", "3gp", "mts", "m2ts"}
_TEXT = {"txt", "jsonl", "ndjson", "csv", "tsv", "md", "srt", "vtt", "conll", "conllu"}
TIMESERIES_MODALITIES = {"imu", "gnss", "can", "biosignal", "tabular", "array"}
TS_COLUMNS = (
    "timestamp",
    "timestamp_ns",
    "timestamp_us",
    "timestamp_ms",
    "timestamp_s",
    "time",
    "time_s",
    "ts",
    "t",
    "unix_time",
    "epoch",
    "sec",
    "seconds",
    "datetime",
    "date_time",
    "header.stamp",
)


def probe_file(loc: Location, rel: str, modality: str) -> Dict[str, Any]:
    """Dispatch on extension, so a radar .npy or an IMU .csv is probed too."""
    ext = extension(rel)
    try:
        if ext in _IMAGE:
            return _probe_image(loc, rel)
        if ext in _AUDIO:
            return _probe_audio(loc, rel)
        if ext in _VIDEO:
            return _probe_av(loc, rel)
        if ext in _TEXT:
            return _probe_text(loc, rel)
        if ext == "npy":
            meta = _probe_npy(loc, rel)
            if modality not in TIMESERIES_MODALITIES:
                meta.pop("records", None)  # a radar cube's first axis isn't a record count
            return meta
        if ext == "parquet":
            return _probe_parquet(loc, rel)
    except Exception as e:  # probes never break a commit
        return {"probe_error": type(e).__name__}
    return {}


def _probe_npy(loc: Location, rel: str) -> Dict[str, Any]:
    """Read only the .npy header (no numpy needed): shape + dtype."""
    import ast

    with loc.open(rel, "rb") as f:
        head = f.read(10)
        if head[:6] != b"\x93NUMPY":
            return {}
        major = head[6]
        hlen = int.from_bytes(head[8:10], "little") if major == 1 else int.from_bytes(head[8:10] + f.read(2), "little")
        if hlen > 65536:  # real headers are < 1 KB; refuse absurd ones
            return {}
        header = f.read(hlen)
    d = ast.literal_eval(header.decode("latin1"))
    shape = list(d.get("shape", ()))
    meta: Dict[str, Any] = {"shape": shape, "dtype": d.get("descr")}
    if shape:
        meta["records"] = shape[0]
    return meta


def _probe_parquet(loc: Location, rel: str) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    with loc.open(rel, "rb") as f:
        md = pq.ParquetFile(f).metadata
        return {"records": md.num_rows, "columns": md.num_columns}


def _to_seconds(values: list, column: str) -> Optional[list]:
    """Parse timestamps (numeric epoch in s/ms/us/ns, or ISO-8601) to float seconds."""
    from datetime import datetime

    out = []
    for v in values:
        v = v.strip()
        try:
            out.append(float(v))
        except ValueError:
            try:
                out.append(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp())
            except ValueError:
                return None
    if not out:
        return None
    c, big = column.lower(), max(abs(out[0]), abs(out[-1]))
    scale = 1.0
    if c.endswith("_ns") or big > 1e17:
        scale = 1e-9
    elif c.endswith("_us") or big > 1e14:
        scale = 1e-6
    elif c.endswith("_ms") or big > 1e11:
        scale = 1e-3
    return [x * scale for x in out]


def _probe_image(loc: Location, rel: str) -> Dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {}
    with loc.open(rel, "rb") as f:
        with Image.open(f) as im:
            return {"width": im.width, "height": im.height, "mode": im.mode}


def _probe_audio(loc: Location, rel: str) -> Dict[str, Any]:
    if extension(rel) == "wav":
        try:
            with loc.open(rel, "rb") as f, wave.open(f) as w:
                frames, rate = w.getnframes(), w.getframerate()
                return {
                    "duration_sec": round(frames / float(rate), 3) if rate else None,
                    "sample_rate": rate,
                    "channels": w.getnchannels(),
                }
        except wave.Error:
            pass  # e.g. float WAV -> fall through to ffprobe
    return _probe_av(loc, rel)


def _probe_av(loc: Location, rel: str) -> Dict[str, Any]:
    path = loc.local_path(rel)
    if not (_FFPROBE and path):
        return {}
    out = subprocess.run(  # fixed argv, no shell, path is a local file  # nosec B603
        [_FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        timeout=60,
    )
    if out.returncode != 0:
        return {"probe_error": "ffprobe"}
    info = json.loads(out.stdout or b"{}")
    meta: Dict[str, Any] = {}
    dur = (info.get("format") or {}).get("duration")
    if dur:
        meta["duration_sec"] = round(float(dur), 3)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and "width" not in meta:
            meta.update(width=s.get("width"), height=s.get("height"), codec=s.get("codec_name"))
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
            n, _, d = rate.partition("/")
            if d and float(d):
                meta["fps"] = round(float(n) / float(d), 3)
            if s.get("nb_frames"):
                meta["frames"] = int(s["nb_frames"])
        elif s.get("codec_type") == "audio" and "sample_rate" not in meta:
            meta.update(sample_rate=int(s.get("sample_rate") or 0) or None, channels=s.get("channels"))
    return meta


def _probe_text(loc: Location, rel: str) -> Dict[str, Any]:
    ext = extension(rel)
    if ext in {"csv", "tsv"}:
        return _probe_csv(loc, rel, "\t" if ext == "tsv" else ",")
    lines = words = 0
    with loc.open(rel, "rb") as f:
        for raw in f:
            if raw.strip():
                lines += 1
                if ext in {"txt", "md"}:
                    words += len(raw.split())
    if ext in {"jsonl", "ndjson"}:
        return {"records": lines}
    return {"lines": lines, "words": words} if words else {"lines": lines}


def count_record_labels(loc: Location, rel: str, column: str) -> Optional[Counter]:
    """Count values of `column` in a csv/tsv/jsonl file (record-level class balance)."""
    ext = extension(rel)
    counts: Counter = Counter()
    try:
        with loc.open(rel, "rb") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="")
            if ext in {"csv", "tsv"}:
                reader = csv.DictReader(text, delimiter="\t" if ext == "tsv" else ",")
                if column not in (reader.fieldnames or []):
                    return None
                for row in reader:
                    counts[str(row.get(column))] += 1
            elif ext in {"jsonl", "ndjson"}:
                for line in text:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and column in obj:
                            counts[str(obj[column])] += 1
            else:
                return None
    except Exception:
        return None
    return counts


def _probe_csv(loc: Location, rel: str, delim: str) -> Dict[str, Any]:
    """Rows/columns, and for sensor logs (IMU, radar, GNSS, CAN...) the time span and
    sample rate taken from a timestamp-like column."""
    with loc.open(rel, "rb") as fb:
        text = io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text, delimiter=delim)
        header = next(reader, None)
        if header is None:
            return {"records": 0}
        cols = [h.strip() for h in header]
        lower = [c.lower() for c in cols]
        ts_idx = next((lower.index(c) for c in TS_COLUMNS if c in lower), None)
        rows, first, last = 0, None, None
        for row in reader:
            if not row:
                continue
            rows += 1
            if ts_idx is not None and ts_idx < len(row):
                if first is None:
                    first = row[ts_idx]
                last = row[ts_idx]
    meta: Dict[str, Any] = {"records": rows, "columns": len(cols)}
    if ts_idx is not None and first is not None and rows > 1:
        secs = _to_seconds([first, last], cols[ts_idx])
        if secs:
            dur = secs[1] - secs[0]
            if dur > 0:
                meta.update(
                    timestamp_column=cols[ts_idx], start=round(secs[0], 6), duration_sec=round(dur, 3), sample_rate_hz=round((rows - 1) / dur, 3)
                )
    return meta
