# Benchmarks: MNIST and Fashion-MNIST

mlfingerprint 0.1.0 on two well-known 70,000-image datasets, each exported to the standard
ImageFolder layout (`train/<class>/*.png`, `test/<class>/*.png`). The raw files were checked against
their published MD5 checksums, and every count mlfingerprint reports was cross-checked against ground
truth computed **independently from the raw IDX pixel arrays** (no PNGs, no mlfingerprint involved).

Machine: 2 vCPUs, Linux-6.18.44-fc-v726-x86_64-with-glibc2.39, Python 3.13.18. Local disk, `reference` mode unless noted.
Reproduce: `bash benchmarks/download.sh raw && python benchmarks/run_benchmark.py --raw raw --work work && python benchmarks/report.py`

## Correctness: mlfingerprint vs ground truth

| Check | MNIST | Fashion-MNIST |
|---|---:|---:|
| Files indexed = 70,000 | ✅ | ✅ |
| Per-split class counts match the label files | ✅ | ✅ |
| Cross-split duplicate count matches pixel-level truth | ✅ | ✅ |
| Total duplicate count matches pixel-level truth | ✅ | ✅ |
| One-pixel edit to one test image caught by `verify` | ✅ | ✅ |
| Flipped copy of a test image placed in train is flagged | ✅ | ✅ |

### What the data contains

| | MNIST | Fashion-MNIST |
|---|---:|---:|
| Train / test images | 60,000 / 10,000 | 60,000 / 10,000 |
| Exact duplicate images inside train | 0 | 0 |
| Exact duplicate images inside test | 0 | 0 |
| Test images whose exact pixels also appear in train | 0 | 0 |
| Classes found from folder names | 10 | 10 |
| Resolution | 28x28 | 28x28 |
| Dataset fingerprint | `sha256:cd7ff2f1f6ae0cf5…` | `sha256:e77fe4b56f0b249f…` |

Both datasets are clean at the **exact**-duplicate level: no image appears twice and no test image is a byte-for-byte
copy of a training image, and mlfingerprint reports exactly that. Exact hashing cannot see **near**-duplicates
(re-encoded, resized or slightly shifted copies). Those are known to exist in other benchmarks, e.g. CIFAR-10/100,
and need perceptual hashing, which is on the roadmap.

## Behaviour on change

| Scenario | MNIST | Fashion-MNIST |
|---|---:|---:|
| One pixel changed in one test PNG → `verify` | FAILED, names `test/1/test_02235.png` | FAILED, names `test/bag/test_02400.png` |
| Re-commit after that edit | `mnist-v1.0.1` (patch bump) | `fashion-v1.0.1` (patch bump) |
| +600 hflip augmentations of train + 1 of a test image | `mnist-v1.1.0` (minor), 601 added | `fashion-v1.1.0` (minor), 601 added |
| …augmentation leak flagged | 1 file: `train/0/test_00525_hflip.png` | 1 file: `train/ankle_boot/test_00470_hflip.png` |
| Content-addressed snapshot v2: objects written / reused | 600 / 70,000 | 600 / 70,000 |

## Speed (wall clock, 70,000 files)

| Operation | MNIST | Fashion-MNIST |
|---|---:|---:|
| First commit (hash everything + probe image sizes) | 19.80 s | 21.45 s |
| Re-commit, nothing changed (hash cache) | 7.10 s | 6.97 s |
| Re-commit after one file changed | 7.59 s | 7.79 s |
| Commit a new batch of 601 files | 8.54 s | 8.36 s |
| `verify` full re-hash | 4.00 s | 3.61 s |
| `verify --sample 0.05` | 0.17 s | 0.40 s |
| `verify --quick` (size only) | 2.42 s | 2.58 s |
| CAS snapshot, first (copies 70k objects) | 24.62 s | 20.25 s |
| CAS snapshot, second (only new objects copied) | 10.84 s | 11.34 s |

## Footprint

| | MNIST | Fashion-MNIST |
|---|---:|---:|
| Data (70,000 PNGs) | 19.4 MB | 35.7 MB |
| `manifest.json` | 3.6 KB | 3.8 KB |
| `index.jsonl.gz` (one row per file) | 4.0 MB | 4.0 MB |

The index costs about 57 bytes per file compressed. For MNIST-sized images (280–510 bytes each) that is a
noticeable 11–20 % of the data. For typical datasets (images and audio of hundreds of KB, video of MBs) it is far
below 0.1 %.

## Honest notes
- Timings come from a 2-vCPU sandbox on local disk. Object stores add network latency per file, and the
  integration tests (`tests/integration`) cover correctness there, not speed.
- A no-change re-commit still walks and `stat`s every file (~100 µs per file here). It doesn't re-hash anything,
  but it isn't free on 10M+ files.
- The datasets were exported to PNG for this benchmark. The hashes describe those PNG files, which is what a
  training pipeline reads. Raw `.gz` files can be versioned directly too.
