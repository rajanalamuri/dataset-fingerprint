"""Modality detection for any sensor stack.

Order of precedence:
  1. user rules        - modality_rules={"radar": ["*/mmwave/**", "*.rdc"]}
  2. sensor folders    - .../radar_front/000123.bin -> "radar", .../imu/accel.csv -> "imu"
                         (only overrides *generic* extensions: .bin/.npy/.csv/.h5/...; a .png in
                          camera/ stays "image", an .mp4 stays "video")
  3. file extension    - .wav -> audio, .mp4 -> video, .png -> image, .pcd -> pointcloud, ...
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

MODALITY_EXTENSIONS = {
    "image": {"jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp", "heic", "heif", "dng", "exr", "ppm", "pgm"},
    "video": {"mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg", "flv", "wmv", "3gp", "mts", "m2ts"},
    "audio": {"wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "opus", "wma", "aiff", "aif", "amr", "pcm"},
    "text": {"txt", "md", "json", "jsonl", "ndjson", "xml", "html", "htm", "srt", "vtt", "conll", "conllu", "yaml", "yml"},
    "tabular": {"csv", "tsv", "parquet", "arrow", "feather", "orc", "avro", "xlsx", "xls"},
    "array": {"npy", "npz", "h5", "hdf5", "pt", "pth", "safetensors", "tfrecord", "pkl", "pickle", "mat", "zarr"},
    "pointcloud": {"pcd", "ply", "las", "laz", "e57"},
    "log": {"bag", "mcap", "db3", "ulg", "bin", "dat", "raw"},
    "archive": {"zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "zst"},
}

# folder keyword -> modality. Matched against each directory name: exact, or as a
# prefix/suffix token ("radar_front", "front-radar", "imu0", "cam_left").
SENSOR_KEYWORDS: Dict[str, Sequence[str]] = {
    "radar": ("radar", "mmwave", "rdr"),
    "lidar": ("lidar", "velodyne", "ouster", "hesai", "livox", "pointcloud", "pointclouds"),
    "imu": ("imu", "accel", "accelerometer", "acc", "gyro", "gyroscope", "magnetometer", "mag", "motion", "inertial"),
    "gnss": ("gps", "gnss", "rtk", "ins"),
    "can": ("can", "canbus", "vehicle_bus", "obd"),
    "biosignal": ("ecg", "eeg", "emg", "ppg", "eda"),
    "depth": ("depth", "tof", "disparity"),
    "thermal": ("thermal", "ir", "flir"),
    "audio": ("audio", "mic", "microphone", "mics"),
    "camera": ("camera", "cam", "rgb", "frames", "images"),
    "event": ("event", "events", "dvs"),
}
# extensions generic enough that a sensor folder should decide the modality
_GENERIC = {"tabular", "array", "log", "other", "pointcloud", "archive"}
# "camera" folders hold images/video already; keep their extension-based modality
_FOLDER_ONLY = {"camera"}

_EXT_TO_MODALITY = {ext: m for m, exts in MODALITY_EXTENSIONS.items() for ext in exts}
_KW_TO_MODALITY = {kw: m for m, kws in SENSOR_KEYWORDS.items() for kw in kws}
_TOKEN = re.compile(r"[-_.\s]+")
_USER_RULES: Dict[str, list] = {}

Rules = Optional[Mapping[str, Union[str, Iterable[str]]]]


def extension(path: str) -> str:
    name = posixpath.basename(path).lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    return name.rsplit(".", 1)[-1] if "." in name else ""


def sensor_keyword(dirname: str) -> Optional[str]:
    """'radar_front' -> 'radar', 'imu0' -> 'imu', 'cam_left' -> 'camera', 'train' -> None."""
    d = dirname.lower()
    if d in _KW_TO_MODALITY:
        return _KW_TO_MODALITY[d]
    tokens = [t for t in _TOKEN.split(d) if t]
    for t in tokens:
        base = t.rstrip("0123456789")
        if base in _KW_TO_MODALITY and (base == t or len(base) >= 2):
            return _KW_TO_MODALITY[base]
    return None


def sensor_dir_index(rel: str) -> Optional[int]:
    """Index of the first directory component that names a sensor."""
    for i, p in enumerate(rel.split("/")[:-1]):
        if sensor_keyword(p):
            return i
    return None


def _match_rules(rel: str, rules: Rules) -> Optional[str]:
    for source in (rules or {}, _USER_RULES):
        for modality, pats in source.items():
            pats = [pats] if isinstance(pats, str) else list(pats)
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(posixpath.basename(rel), p) for p in pats):
                return modality
    return None


def detect_modality(path: str, rules: Rules = None) -> str:
    hit = _match_rules(path, rules)
    if hit:
        return hit
    by_ext = _EXT_TO_MODALITY.get(extension(path), "other")
    if by_ext in _GENERIC:
        for p in reversed(path.split("/")[:-1]):  # nearest folder wins: radar/meta/x.json? -> radar
            kw = sensor_keyword(p)
            if kw and kw not in _FOLDER_ONLY:
                return kw
    return by_ext


def register_extension(ext: str, modality: str) -> None:
    """Teach dataset-fingerprint a custom extension, e.g. register_extension('rdc', 'radar')."""
    ext = ext.lower().lstrip(".")
    MODALITY_EXTENSIONS.setdefault(modality, set()).add(ext)
    _EXT_TO_MODALITY[ext] = modality


def register_sensor(modality: str, *folder_keywords: str) -> None:
    """register_sensor('sonar', 'sonar', 'sss') - folders named like this hold `modality` data."""
    SENSOR_KEYWORDS[modality] = tuple(SENSOR_KEYWORDS.get(modality, ())) + folder_keywords
    for kw in folder_keywords:
        _KW_TO_MODALITY[kw.lower()] = modality


def register_rule(modality: str, *globs: str) -> None:
    """Global glob rule, e.g. register_rule('radar', '*.radar.npy')."""
    _USER_RULES.setdefault(modality, []).extend(globs)
