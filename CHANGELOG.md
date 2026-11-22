# Changelog

## 0.1.0

_Unreleased._
**Core**
- Immutable versions: manifest + per-file index (sha256) + split lists; manifest written last; commit-race protection.
- Storage via fsspec: local, S3 (+MinIO/R2), Azure Blob/ADLS, GCS, Databricks (dbfs:/, /dbfs, /Volumes), HF, HDFS…; parallel bucket listing; ETag/mtime hash cache.
- Modes: reference (zero copy), copy, hardlink, cas (content-addressed, dedup across versions); `checkout` from any mode.
- Auto semantic versioning from the diff; idempotent commits; tags with history.

**Data understanding**
- Modality by extension + sensor folder (radar, lidar, imu, gnss, can, biosignal, depth, thermal, event…); custom rules.
- Splits from dirs/file names (incl. `train2017`, `instances_val2017.json`, HF shards) or stable hash split.
- Leakage-safe auto grouping: recording → sequence → frame block → file; image/annotation pairs; augmented copies.
- Temporal splits (ratios or cutoffs) and row-level time splits for forecasting.
- Probes: audio/video duration, resolution, fps; csv sensor logs (duration, sample rate); npy shape/dtype; parquet rows.
- Labels from class folders or csv/jsonl columns.

**Quality & lineage**
- Checks: augmentation leaks, augmented data in eval, duplicate content across splits, record overlap (NLP/tabular), parent val/test leakage, temporal overlap, empty/unreadable files.
- Parents (other versions or external refs like Delta tables), `trace()`, `check_lineage()` (stale/altered), `load_lock()`.
- `record_training()` across multiple datasets with recipe + seed; PyTorch, MLflow and feature-store helpers.
- CLI: commit, log, show, diff, verify (--sample, --quick), files, lock, tag, tags, lineage, checkout.

**Detection datasets**
- `label_format="yolo"` (+ `class_names`, CLI `--label-format yolo --class-names`) counts objects per class in YOLO label files; box coordinates are never treated as text records.
- Frame-sequence detection requires dense or regularly spaced numbering, so sparse ids (COCO, ImageNet) are no longer grouped as video frames. Fixed-rate timestamp names use their real step for frame blocks.

**Security & quality**
- Path validation for every metadata-supplied path (no absolute paths, `..`, drive letters, NUL); `checkout` refuses an index that doesn't match the manifest; `verify` reports unsafe paths.
- Symlinks that resolve outside the source are never read; hash cache keyed on mtime + ctime + inode (ETag remotely).
- Hardened concurrent commits on object stores; bounded `.npy` header parsing.
- Test suites: unit, security regression, integration on S3 / Azure / GCS emulators; CI with bandit, CodeQL, pip-audit, gitleaks, Scorecard; Trusted-Publishing release with provenance.
