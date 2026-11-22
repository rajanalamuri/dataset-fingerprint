# dataset-fingerprint — POC results

Output of `python examples/poc_multimodal.py /tmp/fp-poc`. The data is synthetic: real MP4s via ffmpeg, real WAV/PNG files, frames extracted at 10 fps, radar `.npy` cubes, 200 Hz IMU CSVs, hourly energy-load CSVs, and NLP jsonl.

```text

==============================================================================
1. Initial release - video + audio + images + NLP captions, train/val/test dirs
==============================================================================
av-multimodal-v1.0.0  (2026-11-21T23:35:47+00:00)
  message     : initial release
  fingerprint : sha256:500a35d283e39d6e25d761296560fecc9ec990baa609acd558044b82af225ab3
  data        : /tmp/fp-poc/av_dataset  [reference]
  files       : 45   size: 478.5 KB
  splits      : train=25 (56%)  val=10 (22%)  test=10 (22%)
  modalities  : audio=12 (6.8s)  image=21  text=3 (30 rec)  video=9 (18.0s)
  classes     : cyclist=34.7%  pedestrian=34.7%  car=30.6%
  transform   : resized frames to 64x48
  transform   : audio -> 16kHz mono
  ! 10 identical file(s) appear in more than one split, e.g. ['test/captions.jsonl', 'val/captions.jsonl']
  ! 5 test record(s), 5 val record(s) also appear in train (exact, normalised) - test contamination

==============================================================================
2. New collection batch arrives -> auto MINOR bump
==============================================================================
av-multimodal-v1.1.0  (2026-11-21T23:35:48+00:00)
  message     : added batch-2026-05-09
  fingerprint : sha256:98ceff46f9774259d9e45247fd9b152bcf5faefb66539aa751b3429108226e85
  data        : /tmp/fp-poc/av_dataset  [reference]
  files       : 87   size: 956.2 KB
  splits      : train=49 (56%)  val=19 (22%)  test=19 (22%)
  modalities  : audio=24 (13.5s)  image=42  text=3 (60 rec)  video=18 (36.0s)
  classes     : cyclist=34.7%  car=32.6%  pedestrian=32.6%
  lineage     : from av-multimodal-v1.0.0 (minor): +42 -0 ~3
  ! 14 identical file(s) appear in more than one split, e.g. ['test/captions.jsonl', 'val/captions.jsonl']
  ! 10 test record(s), 10 val record(s) also appear in train (exact, normalised) - test contamination

==============================================================================
3. Filter corrupt files -> auto PATCH bump
==============================================================================
av-multimodal-v1.1.1  (2026-11-21T23:35:49+00:00)
  message     : removed 2 corrupt frames
  fingerprint : sha256:f509ee4e123cd3ad1e4ef99cf0c2137a331277f9676b7a2712753bc5ed5570cf
  data        : /tmp/fp-poc/av_dataset  [reference]
  files       : 85   size: 938.0 KB
  splits      : train=47 (55%)  val=19 (22%)  test=19 (22%)
  modalities  : audio=24 (13.5s)  image=40  text=3 (60 rec)  video=18 (36.0s)
  classes     : cyclist=35.2%  pedestrian=33.1%  car=31.7%
  lineage     : from av-multimodal-v1.1.0 (patch): +0 -2 ~0
  transform   : {"name": "filter_corrupt_frames", "removed_frames": 2, "reason": "CRC mismatch"}
  ! 14 identical file(s) appear in more than one split, e.g. ['test/captions.jsonl', 'val/captions.jsonl']
  ! 10 test record(s), 10 val record(s) also appear in train (exact, normalised) - test contamination

==============================================================================
4. Same data again -> nothing to commit (idempotent, CI-safe)
==============================================================================
returned existing av-multimodal-v1.1.1; versions on disk: ['av-multimodal-v1.0.0', 'av-multimodal-v1.1.0', 'av-multimodal-v1.1.1']

==============================================================================
5. New sensor modality (lidar .pcd) -> auto MAJOR bump
==============================================================================
av-multimodal-v2.0.0  (2026-11-21T23:35:49+00:00)
  message     : added lidar sweeps (new collection rig)
  fingerprint : sha256:9d03ca7459476ddbe1e419c46e8ed79368c52a7f54058713827dde8c678a8096
  data        : /tmp/fp-poc/av_dataset  [reference]
  files       : 94   size: 938.2 KB
  splits      : train=50 (53%)  val=22 (23%)  test=22 (23%)
  modalities  : audio=24 (13.5s)  image=40  lidar=9  text=3 (60 rec)  video=18 (36.0s)
  classes     : cyclist=35.2%  pedestrian=33.1%  car=31.7%
  lineage     : from av-multimodal-v1.1.1 (major): +9 -0 ~0
  ! 17 identical file(s) appear in more than one split, e.g. ['test/captions.jsonl', 'val/captions.jsonl']
  ! 10 test record(s), 10 val record(s) also appear in train (exact, normalised) - test contamination

==============================================================================
6. Why is v2.0.0 different from v1.0.0?  -> diff
==============================================================================
av-multimodal-v1.0.0 -> av-multimodal-v2.0.0
  +49 added  -0 removed  ~3 modified  ↻0 renamed  ⇄0 split-changed
  + [test] test/audio/car/b2_0.wav
  + [test] test/audio/cyclist/b2_0.wav
  + [test] test/audio/pedestrian/b2_0.wav
  + [test] test/images/car/b2_0.png
  + ... 45 more
  ~ [test] test/captions.jsonl
  ~ [train] train/captions.jsonl
  ~ [val] val/captions.jsonl
  new modalities: lidar
  label delta: car:+12, cyclist:+14, pedestrian:+14

==============================================================================
7. Training job: load by version, use files, record the model
==============================================================================
av-multimodal-v1.1.1: 47 train samples, e.g. {"uri": "/tmp/fp-poc/av_dataset/train/audio/car/b1_0.wav", "label": "car", "modality": "audio", "split": "train", "channels": 1, "duration_sec": 0.5, ...
model metadata to store with the model:
{
  "dataset": "av-multimodal",
  "dataset_version": "av-multimodal-v1.1.1",
  "fingerprint": "sha256:f509ee4e123cd3ad1e4ef99cf0c2137a331277f9676b7a2712753bc5ed5570cf",
  "manifest_uri": "/tmp/fp-poc/ledger/av-multimodal-v1.1.1/manifest.json",
  "split_checksums": {
    "train_split": "sha256:c015ccb963f7ac27c94ede8b193afdccc69017eaf4c69f81b5d7e5806d92d8ba",
    "val_split": "sha256:84081c569ae9726b0c9d80e8c9dc92f550380856f8356f243a2e9963000db059",
    "test_split": "sha256:e659fa60c9f7a3a4f075cd67c02cbb855a27878c01b638bd6bedd4f273f4ad07"
  }
}

==============================================================================
8. Someone edits a training file in place -> verify catches it
==============================================================================
verify av-multimodal-v1.1.1: FAILED  (84/85 files intact)
  modified : val/audio/pedestrian/b1_0.wav

==============================================================================
9. Unsplit speech corpus: speaker-grouped stable hash split (no speaker leakage)
==============================================================================
speech-v1.0.0  (2026-11-21T23:35:49+00:00)
  message     : -
  fingerprint : sha256:1ab632df34a2a731a60bd6e37053b568e398ad9912e9846b72b9cf86683379c2
  data        : /tmp/fp-poc/speech_raw  [reference]
  files       : 40   size: 376.7 KB
  splits      : train=32 (80%)  val=4 (10%)  test=4 (10%)
  modalities  : audio=40 (12.0s)
speakers in >1 split: none

==============================================================================
10. Multi-sensor drives: camera frames @10fps + radar + IMU + mic, no splits given
==============================================================================
drives-v1.0.0  (2026-11-21T23:35:53+00:00)
  message     : 24 drive recordings
  fingerprint : sha256:17d6da1dc8c0b438da560b3570c392dd72113c1f975df8e4e91c2d76b420975f
  data        : /tmp/fp-poc/drives_raw  [reference]
  files       : 528   size: 9.7 MB
  splits      : train=462 (88%)  val=22 (4%)  test=44 (8%)
  modalities  : audio=24 (24.0s)  image=240  imu=24 (23.9s) (4,800 rec)  radar=240
  ! 22 identical file(s) appear in more than one split, e.g. ['2026-05-01_run00/camera_front/000001.jpg', '2026-05-10_run09/camera_front/000001.jpg']
split strategy: {"type": "hash", "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "group_by": "auto:recording", "seed": "dataset-fingerprint", "groups": 24}
sequences     : {"image": {"sequences": 24, "frames": 240, "min_frames": 10, "max_frames": 10}, "radar": {"sequences": 24, "frames": 240, "min_frames": 10, "max_frames": 10}}
imu file meta : {"columns": 7, "duration_sec": 0.995, "records": 200, "sample_rate_hz": 200.0, "start": 1715013000.0, "timestamp_column": "timestamp_ns"}
radar meta    : {"dtype": "<f4", "shape": [64, 128]}
recordings split across >1 split: none

==============================================================================
11. Offline augmentation -> derived dataset with parents, leak checks, lineage
==============================================================================
av-aug-v1.0.0  (2026-11-21T23:35:53+00:00)
  message     : -
  fingerprint : sha256:db255b7f9d31308cf2ab3398d271f5e5c99948fa7545057865f5e5354b6cf667
  data        : /tmp/fp-poc/av_aug  [reference]
  files       : 29   size: 264.3 KB
  splits      : train=29 (100%)
  modalities  : image=29
  classes     : cyclist=34.5%  pedestrian=34.5%  car=31.0%
  parent      : av-multimodal-v2.0.0
  augmented   : 29 files  train=29
  transform   : {"name": "hflip", "p": 1.0}
  ! 1 training file(s) come from a parent's val/test split, e.g. train/images/car/b2_0_hflip.png augments a av-multimodal-v2.0.0 val/test file
{
  "dataset_version": "av-aug-v1.0.0",
  "fingerprint": "sha256:db255b7f9d31308cf2ab3398d271f5e5c99948fa7545057865f5e5354b6cf667",
  "message": "",
  "parents": [
    {
      "dataset_version": "av-multimodal-v2.0.0",
      "fingerprint": "sha256:9d03ca7459476ddbe1e419c46e8ed79368c52a7f54058713827dde8c678a8096",
      "message": "added lidar sweeps (new collection rig)",
      "parents": []
    }

==============================================================================
12. One training run on several datasets + on-the-fly augmentation recipe
==============================================================================
{"model_version": "model-v2.0.0", "recipe": {"random_crop": 0.5, "color_jitter": 0.2, "specaugment": true}, "seed": 1234}
datasets: ['av-multimodal-v2.0.0', 'av-aug-v1.0.0']
reopen exact data from the stored lock: <DatasetVersion av-multimodal-v2.0.0 files=94 splits=['train', 'val', 'test'] sha256:9d03ca745947>

==============================================================================
13. Time-series forecasting: temporal split (past -> train, future -> test)
==============================================================================
energy-v1.0.0  (2026-11-21T23:35:53+00:00)
  message     : -
  fingerprint : sha256:8908afc11806f28f76e9290bb025bb0d98cb505a03a2f79ae9cd0f5c97b830a2
  data        : /tmp/fp-poc/energy_load  [reference]
  files       : 30   size: 18.8 KB
  splits      : train=21 (70%)  val=5 (17%)  test=4 (13%)
  modalities  : tabular=30 (690.0h) (720 rec)
time ranges: {"train": {"start": "2026-04-01T00:00:00+00:00", "end": "2026-04-21T23:00:00+00:00"}, "val": {"start": "2026-04-22T00:00:00+00:00", "end": "2026-04-26T23:00:00+00:00"}, "test": {"start": "2026-04-27T00:00:00+00:00", "end": "2026-04-30T23:00:00+00:00"}}

==============================================================================
14. NLP: test contamination (same sentence in train and test)
==============================================================================
quality: {"record_overlap_with_train": {"test": 1}}
warnings: ['1 test record(s) also appear in train (exact, normalised) - test contamination']

==============================================================================
15. Tags + content-addressed snapshots + checkout
==============================================================================
prod -> av-multimodal-v1.1.1
av-v1.1.0: objects written 1, reused 45 (dedup)
checkout -> /tmp/fp-poc/checkout_v1

==============================================================================
16. Ledger history
==============================================================================
VERSION                     CREATED                        FILES  BUMP     MESSAGE
av-multimodal-v2.0.0        2026-11-21T23:35:49+00:00         94  major    added lidar sweeps (new collection rig)
av-multimodal-v1.1.1        2026-11-21T23:35:49+00:00         85  patch    removed 2 corrupt frames
av-multimodal-v1.1.0        2026-11-21T23:35:48+00:00         87  minor    added batch-2026-05-09
av-multimodal-v1.0.0        2026-11-21T23:35:47+00:00         45  initial  initial release

manifest: /tmp/fp-poc/ledger/av-multimodal-v2.0.0/manifest.json
```

## Cloud emulators (`examples/cloud_emulator_check.py`)

```text
S3 temporal+cas : drives-v1.0.1 {'start': '2026-04-23T00:00:00+00:00', 'end': '2026-04-24T00:00:00+00:00'} objects: 144
S3 verify       : verify drives-v1.0.1: OK  (144/144 files intact) | quick: True
S3 checkout     : s3://fp-ledger/checkout-test
S3 tamper (ref) : caught = True ['2026-04-01_run0/radar/000000.bin']
S3 cas snapshot : still intact = True
S3 -> Azure     : drives-aug-v1.0.1 parent: drives-v1.0.1 verify: True
abfs://         : dataset-v1.0.1
```

S3 via moto server and Azure Blob via Azurite. Databricks `/Volumes` and `/dbfs` use the same code path as local disk. `dbfs:/` off-cluster has not been run against a live workspace.
