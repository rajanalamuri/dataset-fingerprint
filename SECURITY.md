# Security policy

## Reporting a vulnerability
Please **do not open a public issue**. Use GitHub's *Report a vulnerability* button (Security → Advisories)
on this repository. You will get an acknowledgement within 5 working days and a fix or mitigation plan within 30.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Threat model

dataset-fingerprint reads datasets and writes small metadata files (a *ledger*). It never executes data, never
uses `pickle`/`eval`, never opens a shell, and never makes network calls of its own: all I/O goes through
the storage driver you configure (local disk or fsspec).

| Actor | What they can do | What dataset-fingerprint guarantees |
|---|---|---|
| Can modify **data** after a commit | change, delete, add files | `verify` detects every changed/missing file (SHA-256); the fingerprint of the version is unaffected and still identifies the original bytes |
| Can modify **the ledger** (index/manifest) | rewrite metadata | `checkout` refuses an index that no longer matches the manifest's split checksums; paths are validated (no absolute paths, `..`, drive letters, NUL) so a malicious index can never read or write outside its root; `load_lock()` refuses a version whose fingerprint differs from the one a model stored |
| Plants **symlinks** in a source folder | point at files outside the dataset | symlinks that resolve outside the source root are skipped (and warned about), never read or copied |
| Supplies **malformed files** | corrupt WAV/PNG/NPY/CSV, oversized headers | probes fail soft and bound what they read (e.g. `.npy` headers > 64 KB are ignored) |
| Edits files but **restores mtime** | `rsync -t`, `cp -p`, `tar` | the hash cache keys on mtime **and** ctime + inode (ETag on object stores), so content changes are re-hashed |

Credentials passed as `storage_options` are used for I/O only and are never written to manifests, indexes or logs.

## Hardening recommendations
- Give training jobs **read-only** access to the ledger and data; only the pipeline that commits needs write.
- Enable **S3 Object Lock / bucket versioning** (or Azure immutable blob storage) on raw-data prefixes in `reference` mode.
- Store `ds.lock()` with every model (MLflow, model registry). It is the independent anchor that proves a ledger was not rewritten.
- Run `fingerprint verify --sample 0.05` as a CI gate before training; `verify` exits non-zero on any problem.
- Run one committer per dataset (e.g. a CI `concurrency` group). Concurrent commits are guarded, but object stores
  have no atomic create through fsspec, so the guard is best-effort there.

## What CI checks on every change
Bandit (SAST) · CodeQL (security-extended) · pip-audit against all optional dependencies · gitleaks secret scan ·
dependency review on PRs · OpenSSF Scorecard · `tests/security` regression suite (path traversal, symlink escape,
tamper detection, secret leakage, banned calls). Releases are built once in CI, signed with build provenance
attestations and published through PyPI Trusted Publishing (no long-lived tokens).
