# Benchmarks: MNIST, Fashion-MNIST, COCO 2017 and CIFAR-10

## MNIST and Fashion-MNIST

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
and are caught by the opt-in perceptual check (`near_duplicates=True`); see the CIFAR-10 section below.

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
| First commit (hash everything + probe image sizes) | 23.34 s | 22.88 s |
| Re-commit, nothing changed (hash cache) | 8.05 s | 7.16 s |
| Re-commit after one file changed | 9.40 s | 8.62 s |
| Commit a new batch of 601 files | 9.43 s | 10.13 s |
| `verify` full re-hash | 5.65 s | 5.50 s |
| `verify --sample 0.05` | 0.24 s | 0.38 s |
| `verify --quick` (size only) | 2.38 s | 2.75 s |
| CAS snapshot, first (copies 70k objects) | 28.03 s | 25.52 s |
| CAS snapshot, second (only new objects copied) | 13.02 s | 12.52 s |

## Footprint

| | MNIST | Fashion-MNIST |
|---|---:|---:|
| Data (70,000 PNGs) | 19.4 MB | 35.7 MB |
| `manifest.json` | 3.4 KB | 3.6 KB |
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

---

# Benchmarks: COCO 2017

COCO's own servers and Hugging Face are unreachable from the benchmark sandbox, so this uses the two COCO 2017
packages that Ultralytics redistributes on GitHub (`bash benchmarks/download_coco.sh`):

- **Part A**: the complete COCO 2017 annotation set: 117,266 train + 4,952 val YOLO label
  files (every annotated image), the image lists, and the COCO Consortium's official `instances_val2017.json`.
  **No image pixels.**
- **Part B**: `coco128`, 128 real COCO train2017 JPEGs with their label files.

Reproduce: `bash benchmarks/download_coco.sh .bench/coco && python benchmarks/run_coco.py --root .bench/coco --work .bench/coco-work`

## Part A: 122,224 annotation files

| Check (ground truth computed without mlfingerprint) | |
|---|:-:|
| label files per split match | ✅ |
| object counts match train | ✅ |
| object counts match val | ✅ |
| val labels match official noncrowd | ✅ |
| duplicate count matches | ✅ |
| cross split duplicates match | ✅ |
| tamper detected | ✅ |
| copied val label in train flagged | ✅ |

| Finding | Value |
|---|---:|
| Files per split, detected from `train2017/` and `val2017/` (label files + list/annotation files) | 117,267 train · 4,954 val |
| `instances_val2017.json` and `test-dev2017.txt` assigned to | val · test |
| Objects (YOLO boxes), train / val | 849,947 / 36,335 |
| Classes found | 80 |
| Most frequent classes in train | person 257,252, car 43,533, chair 38,073, book 24,077, bottle 24,070 |
| Official val annotations: total / crowd / non-crowd | 36,781 / 446 / 36,335 |
| Listed images with **no** annotation file (train / val) | 1,021 / 48 |
| Byte-identical label files (all inside train, none across splits) | 3 |
| Dataset fingerprint | `sha256:a480bd1d8b1f93f3…` |

The Ultralytics YOLO conversion of val2017 holds exactly the official **non-crowd** objects: crowd regions are
dropped by design, and every remaining object is accounted for, class by class.

The three identical label files are pairs of different images whose only label is one box covering the whole frame
(`train2017/000000499402.txt = train2017/000000517070.txt; train2017/000000220834.txt = train2017/000000295776.txt; train2017/000000415874.txt = train2017/000000485616.txt`). They are not duplicate images. Identical annotations are only a hint; proving a duplicate takes the
pixels.

| Scenario | Result |
|---|---|
| One box coordinate edited in one val label → `verify` | FAILED, names `labels/val2017/000000009914.txt` |
| Re-commit after that edit | `coco2017-annotations-v1.0.1` (patch bump) |
| A val label file copied into train under a new name | flagged: 1 identical file across splits |

| Operation (122,224 files, 2 vCPUs) | Time |
|---|---:|
| First commit (hash + count 886k objects by class) | 35.1 s |
| Re-commit, nothing changed | 23.4 s |
| Re-commit after one file changed | 27.2 s |
| `verify` full / 5 % sample / quick | 9.9 s / 0.88 s / 5.4 s |
| `manifest.json` / `index.jsonl.gz` vs data | 9.3 KB / 8.0 MB vs 59.4 MB |

## Part B: coco128 (real images)

| Check | |
|---|:-:|
| every image label pair in one split | ✅ |
| resolutions match | ✅ |
| one flipped byte in a jpeg detected | ✅ |

128 JPEGs in 58 different resolutions (most common: 640x480 ×35, 640x427 ×12, 480x640 ×9),
all read correctly by the probe. With no splits given, mlfingerprint hash-split the folder train 193, val 38, test 27 files and kept
every image in the same split as its label file. Flipping one byte inside one JPEG was caught by `verify`.

**Found and fixed by this run:** COCO names images by sparse numeric ids (`000000000009.jpg`, `000000000025.jpg`,
…). The first version of mlfingerprint mistook that numbering for video frames and grouped unrelated photos into
blocks, which skewed the split (201 / 17 / 40). Frame sequences now have to be densely or regularly numbered (frame
counters, fixed-rate timestamps). Irregular ids are treated as independent samples, and a regression test covers it.

**Not covered:** the COCO pixels (train2017 18 GB, val2017 1 GB) could not be downloaded here. Committing a COCO folder
that includes `images/` works the same way and reports exact duplicate images across splits, but this benchmark's
independent ground-truth checks cover the annotations only.

# CIFAR-10: near-duplicates across splits (reproducing ciFAIR)

Barz & Denzler (ciFAIR) found **286** CIFAR-10 test images with a near-duplicate in the training set,
using CNN features plus manual review. mlfingerprint has no CNN: `commit(..., near_duplicates=T)` computes a 64-bit
difference hash (dHash) per image and runs an exact Hamming search across splits. Exact byte-level duplicates across
splits: **0**.

| dHash distance | Test images flagged | In the ciFAIR list | Recall vs ciFAIR | Search time |
|---|---:|---:|---:|---:|
| ≤ 0 | 1 | 0 | 0% | 0.0 s |
| ≤ 2 | 34 | 22 | 8% | 0.2 s |
| ≤ 4 | 91 | 63 | 22% | 1.2 s |
| ≤ 6 | 169 | 102 | 36% | 8.6 s |
| ≤ 8 | 360 | 153 | 54% | 38.7 s |
| ≤ 10 | 834 | 194 | 68% | 119.6 s |

**At the default threshold (≤ 4)**, 91 test images are flagged. 63 are in the ciFAIR list.
The other 28 were reviewed by eye ([pairs](results/cifar_unlisted_review.csv),
[contact sheet](results/cifar_unlisted_T4.png)): **24 are clearly the same photograph** as a training
image (colour, contrast or grayscale changes) and 4 are different photos. So about
**87/91 (96%) of flags are real leaks**, and those
24 are not in ciFAIR's list, so they are still in the ciFAIR-10 test set (its kept test images are
pixel-identical to CIFAR-10's).

**Limits, stated plainly:** a 64-bit hash of a 32×32 thumbnail recovers only 22% of ciFAIR's list at ≤ 4 and
54% at ≤ 8, where precision drops. It finds re-encoded and colour-shifted copies, not crops or re-shoots.
It is a cheap guard built into every commit; it does not replace an embedding search. One reviewer; the review is in the repo
so it can be checked.

| Cost (60,000 PNGs) | Time |
|---|---:|
| Commit without the check | 7.5 s |
| Commit with the check (cold) | 35.8 s |
| Re-commit with the check (hashes cached by sha256) | 10.6 s |

Data: CIFAR-10 binary batches (sha256 pinned in `run_cifar.py`); the test batch was cross-checked against ciFAIR-10.
Reproduce: `bash benchmarks/download_cifar.sh .bench/cifar10 && python benchmarks/run_cifar.py --root .bench/cifar10 --work .bench/cifar-work`.
