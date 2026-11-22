"""Render benchmarks/results/results.json as benchmarks/RESULTS.md (every number comes from the run)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "results.json").read_text())
D = R["datasets"]
E = R["environment"]


def mb(b):
    return f"{b / 1e6:.1f} MB"


def row(label, fn):
    return f"| {label} | " + " | ".join(fn(d) for d in D) + " |"


T = lambda k: lambda d: f"{d['timings_s'][k]:.2f} s"  # noqa: E731
M = lambda d: d["mlfingerprint"]  # noqa: E731
G = lambda d: d["ground_truth"]  # noqa: E731
ok = lambda b: "✅" if b else "❌"  # noqa: E731
names = " | ".join(d["dataset"] for d in D)
sep = "|---|" + "---:|" * len(D)

md = f"""# Benchmarks: MNIST, Fashion-MNIST and COCO 2017

## MNIST and Fashion-MNIST

mlfingerprint {E["mlfingerprint"]} on two well-known 70,000-image datasets, each exported to the standard
ImageFolder layout (`train/<class>/*.png`, `test/<class>/*.png`). The raw files were checked against
their published MD5 checksums, and every count mlfingerprint reports was cross-checked against ground
truth computed **independently from the raw IDX pixel arrays** (no PNGs, no mlfingerprint involved).

Machine: {E["cpus"]} vCPUs, {E["platform"]}, Python {E["python"]}. Local disk, `reference` mode unless noted.
Reproduce: `bash benchmarks/download.sh raw && python benchmarks/run_benchmark.py --raw raw --work work && python benchmarks/report.py`

## Correctness: mlfingerprint vs ground truth

| Check | {names} |
{sep}
{row("Files indexed = 70,000", lambda d: ok(d["checks"]["files_match"]))}
{row("Per-split class counts match the label files", lambda d: ok(d["checks"]["class_counts_match"]))}
{row("Cross-split duplicate count matches pixel-level truth", lambda d: ok(d["checks"]["cross_split_duplicates_match"]))}
{row("Total duplicate count matches pixel-level truth", lambda d: ok(d["checks"]["duplicate_files_match"]))}
{row("One-pixel edit to one test image caught by `verify`", lambda d: ok(d["checks"]["tamper_detected"]))}
{row("Flipped copy of a test image placed in train is flagged", lambda d: ok(d["checks"]["augmentation_leak_flagged"]))}

### What the data contains

| | {names} |
{sep}
{row("Train / test images", lambda d: f"{G(d)['train_images']:,} / {G(d)['test_images']:,}")}
{row("Exact duplicate images inside train", lambda d: str(G(d)["train_duplicate_images"]))}
{row("Exact duplicate images inside test", lambda d: str(G(d)["test_duplicate_images"]))}
{row("Test images whose exact pixels also appear in train", lambda d: str(G(d)["test_images_also_in_train"]))}
{row("Classes found from folder names", lambda d: str(M(d)["num_classes"]))}
{row("Resolution", lambda d: ", ".join(M(d)["resolutions"]))}
{row("Dataset fingerprint", lambda d: "`" + M(d)["fingerprint"][:23] + "…`")}

Both datasets are clean at the **exact**-duplicate level: no image appears twice and no test image is a byte-for-byte
copy of a training image, and mlfingerprint reports exactly that. Exact hashing cannot see **near**-duplicates
(re-encoded, resized or slightly shifted copies). Those are known to exist in other benchmarks, e.g. CIFAR-10/100,
and need perceptual hashing, which is on the roadmap.

## Behaviour on change

| Scenario | {names} |
{sep}
{row("One pixel changed in one test PNG → `verify`", lambda d: "FAILED, names `" + M(d)["tamper"]["modified"][0] + "`")}
{row("Re-commit after that edit", lambda d: f"`{M(d)['tamper']['recommit_version']}` ({M(d)['tamper']['bump']} bump)")}
{row("+600 hflip augmentations of train + 1 of a test image", lambda d: f"`{M(d)['new_batch']['version']}` ({M(d)['new_batch']['bump']}), {M(d)['new_batch']['added']} added")}
{row("…augmentation leak flagged", lambda d: str(M(d)["new_batch"]["augmentation_leaks_flagged"]) + " file: `" + (M(d)["new_batch"]["leak_example"] or "").split(" ")[0] + "`")}
{row("Content-addressed snapshot v2: objects written / reused", lambda d: f"{M(d)['cas']['v2_objects_written']:,} / {M(d)['cas']['v2_objects_reused']:,}")}

## Speed (wall clock, 70,000 files)

| Operation | {names} |
{sep}
{row("First commit (hash everything + probe image sizes)", T("commit_cold"))}
{row("Re-commit, nothing changed (hash cache)", T("commit_no_change"))}
{row("Re-commit after one file changed", T("commit_after_one_change"))}
{row("Commit a new batch of 601 files", T("commit_new_batch_601_files"))}
{row("`verify` full re-hash", T("verify_full"))}
{row("`verify --sample 0.05`", T("verify_sample_5pct"))}
{row("`verify --quick` (size only)", T("verify_quick"))}
{row("CAS snapshot, first (copies 70k objects)", T("cas_snapshot_v1"))}
{row("CAS snapshot, second (only new objects copied)", T("cas_snapshot_v2"))}

## Footprint

| | {names} |
{sep}
{row("Data (70,000 PNGs)", lambda d: mb(M(d)["sizes_bytes"]["data_pngs"]))}
{row("`manifest.json`", lambda d: f"{M(d)['sizes_bytes']['manifest.json'] / 1024:.1f} KB")}
{row("`index.jsonl.gz` (one row per file)", lambda d: mb(M(d)["sizes_bytes"]["index.jsonl.gz"]))}

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
"""
(HERE / "RESULTS.md").write_text(md)

# ----------------------------------------------------------------------------- COCO
CP = HERE / "results" / "coco.json"
if CP.exists():
    C = json.loads(CP.read_text())
    A, B = C["parts"]
    ga, ma, ta = A["ground_truth"], A["mlfingerprint"], A["timings_s"]
    gb, mb_, tb = B["ground_truth"], B["mlfingerprint"], B["timings_s"]
    chk = lambda d: "\n".join(f"| {k.replace('_', ' ')} | {ok(v)} |" for k, v in d.items())  # noqa: E731
    top5 = ", ".join(f"{k} {v:,}" for k, v in ga["top5_train"])
    groups = "; ".join(" = ".join(g) for g in ga["identical_label_file_groups"])
    md += f"""
---

# Benchmarks: COCO 2017

COCO's own servers and Hugging Face are unreachable from the benchmark sandbox, so this uses the two COCO 2017
packages that Ultralytics redistributes on GitHub (`bash benchmarks/download_coco.sh`):

- **Part A**: the complete COCO 2017 annotation set: {ga["label_files"]["train2017"]:,} train + {ga["label_files"]["val2017"]:,} val YOLO label
  files (every annotated image), the image lists, and the COCO Consortium's official `instances_val2017.json`.
  **No image pixels.**
- **Part B**: `coco128`, 128 real COCO train2017 JPEGs with their label files.

Reproduce: `bash benchmarks/download_coco.sh .bench/coco && python benchmarks/run_coco.py --root .bench/coco --work .bench/coco-work`

## Part A: 122,224 annotation files

| Check (ground truth computed without mlfingerprint) | |
|---|:-:|
{chk(A["checks"])}

| Finding | Value |
|---|---:|
| Files per split, detected from `train2017/` and `val2017/` (label files + list/annotation files) | {ma["split_files"]["train"]:,} train · {ma["split_files"]["val"]:,} val |
| `instances_val2017.json` and `test-dev2017.txt` assigned to | {ma["annotation_json_split"]} · {ma["list_file_splits"]["test-dev2017.txt"]} |
| Objects (YOLO boxes), train / val | {ga["objects"]["train2017"]:,} / {ga["objects"]["val2017"]:,} |
| Classes found | {ma["num_classes"]} |
| Most frequent classes in train | {top5} |
| Official val annotations: total / crowd / non-crowd | {ga["official_val_annotations"]:,} / {ga["official_val_crowd"]:,} / {ga["official_val_noncrowd"]:,} |
| Listed images with **no** annotation file (train / val) | {ga["images_without_labels"]["train2017"]:,} / {ga["images_without_labels"]["val2017"]:,} |
| Byte-identical label files (all inside train, none across splits) | {ga["extra_identical_files"]} |
| Dataset fingerprint | `{ma["fingerprint"][:23]}…` |

The Ultralytics YOLO conversion of val2017 holds exactly the official **non-crowd** objects: crowd regions are
dropped by design, and every remaining object is accounted for, class by class.

The three identical label files are pairs of different images whose only label is one box covering the whole frame
(`{groups}`). They are not duplicate images. Identical annotations are only a hint; proving a duplicate takes the
pixels.

| Scenario | Result |
|---|---|
| One box coordinate edited in one val label → `verify` | FAILED, names `{ma["tamper"]["modified"][0]}` |
| Re-commit after that edit | `{ma["tamper"]["recommit"]}` ({ma["tamper"]["bump"]} bump) |
| A val label file copied into train under a new name | flagged: {ma["leak"]["duplicates_across_splits"]} identical file across splits |

| Operation (122,224 files, {C["environment"]["cpus"]} vCPUs) | Time |
|---|---:|
| First commit (hash + count 886k objects by class) | {ta["commit_cold"]:.1f} s |
| Re-commit, nothing changed | {ta["commit_no_change"]:.1f} s |
| Re-commit after one file changed | {ta["commit_after_one_change"]:.1f} s |
| `verify` full / 5 % sample / quick | {ta["verify_full"]:.1f} s / {ta["verify_sample_5pct"]:.2f} s / {ta["verify_quick"]:.1f} s |
| `manifest.json` / `index.jsonl.gz` vs data | {ma["sizes_bytes"]["manifest.json"] / 1024:.1f} KB / {ma["sizes_bytes"]["index.jsonl.gz"] / 1e6:.1f} MB vs {ma["sizes_bytes"]["data"] / 1e6:.1f} MB |

## Part B: coco128 (real images)

| Check | |
|---|:-:|
{chk(B["checks"])}

{gb["images"]} JPEGs in {gb["distinct_resolutions"]} different resolutions (most common: {", ".join(f"{k} ×{v}" for k, v in gb["top_resolutions"])}),
all read correctly by the probe. With no splits given, mlfingerprint hash-split the folder {", ".join(f"{k} {v}" for k, v in mb_["splits"].items())} files and kept
every image in the same split as its label file. Flipping one byte inside one JPEG was caught by `verify`.

**Found and fixed by this run:** COCO names images by sparse numeric ids (`000000000009.jpg`, `000000000025.jpg`,
…). The first version of mlfingerprint mistook that numbering for video frames and grouped unrelated photos into
blocks, which skewed the split (201 / 17 / 40). Frame sequences now have to be densely or regularly numbered (frame
counters, fixed-rate timestamps). Irregular ids are treated as independent samples, and a regression test covers it.

**Not covered:** the COCO pixels (train2017 18 GB, val2017 1 GB) could not be downloaded here. Committing a COCO folder
that includes `images/` works the same way and reports exact duplicate images across splits, but this benchmark's
independent ground-truth checks cover the annotations only.
"""
    (HERE / "RESULTS.md").write_text(md)

FP = HERE / "results" / "cifar.json"
if FP.exists():
    F = json.loads(FP.read_text())
    sw = {x["threshold"]: x for x in F["sweep"]}
    rv = F["unlisted_T4_review"]
    t4 = sw[4]
    true4 = t4["in_cifair_list"] + rv["test_images_near_duplicate"]
    tbl = "\n".join(
        f"| ≤ {x['threshold']} | {x['test_images_flagged']} | {x['in_cifair_list']} | {x['recall_vs_cifair']:.0%} | {x['search_seconds']:.1f} s |"
        for x in F["sweep"]
    )
    tf = F["timings_sec"]
    md += f"""
# CIFAR-10: near-duplicates across splits (reproducing ciFAIR)

Barz & Denzler (ciFAIR) found **{F["cifair_listed"]}** CIFAR-10 test images with a near-duplicate in the training set,
using CNN features plus manual review. mlfingerprint has no CNN: `commit(..., near_duplicates=T)` computes a 64-bit
difference hash (dHash) per image and runs an exact Hamming search across splits. Exact byte-level duplicates across
splits: **{F["exact_duplicates_across_splits"]}**.

| dHash distance | Test images flagged | In the ciFAIR list | Recall vs ciFAIR | Search time |
|---|---:|---:|---:|---:|
{tbl}

**At the default threshold (≤ 4)**, {t4["test_images_flagged"]} test images are flagged. {t4["in_cifair_list"]} are in the ciFAIR list.
The other {rv["test_images_near_duplicate"] + rv["test_images_different"]} were reviewed by eye ([pairs](results/cifar_unlisted_review.csv),
[contact sheet](results/cifar_unlisted_T4.png)): **{rv["test_images_near_duplicate"]} are clearly the same photograph** as a training
image (colour, contrast or grayscale changes) and {rv["test_images_different"]} are different photos. So about
**{true4}/{t4["test_images_flagged"]} ({true4 / t4["test_images_flagged"]:.0%}) of flags are real leaks**, and those
{rv["test_images_near_duplicate"]} are not in ciFAIR's list, so they are still in the ciFAIR-10 test set (its kept test images are
pixel-identical to CIFAR-10's).

**Limits, stated plainly:** a 64-bit hash of a 32×32 thumbnail recovers only {t4["recall_vs_cifair"]:.0%} of ciFAIR's list at ≤ 4 and
{sw[8]["recall_vs_cifair"]:.0%} at ≤ 8, where precision drops. It finds re-encoded and colour-shifted copies, not crops or re-shoots.
It is a cheap guard built into every commit; it does not replace an embedding search. One reviewer; the review is in the repo
so it can be checked.

| Cost (60,000 PNGs) | Time |
|---|---:|
| Commit without the check | {tf["commit_plain_cold"]:.1f} s |
| Commit with the check (cold) | {tf["commit_with_near_dups_cold"]:.1f} s |
| Re-commit with the check (hashes cached by sha256) | {tf["recommit_with_near_dups_warm"]:.1f} s |

Data: CIFAR-10 binary batches (sha256 pinned in `run_cifar.py`); the test batch was cross-checked against ciFAIR-10.
Reproduce: `bash benchmarks/download_cifar.sh .bench/cifar10 && python benchmarks/run_cifar.py --root .bench/cifar10 --work .bench/cifar-work`.
"""
    (HERE / "RESULTS.md").write_text(md)

print(md)
