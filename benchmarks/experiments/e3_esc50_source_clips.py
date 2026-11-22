"""E3: source-clip leakage in ESC-50 (2,000 5-s clips; several 'takes' can be cut from one Freesound
recording). Random file split vs split grouped by source clip; and what dataset-fingerprint does."""

import os

DATA = os.environ.get("BENCH_DATA", ".bench")  # set by download_experiments.sh

import json, os, re, shutil, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dataset_fingerprint as fp

A = Path(f"{DATA}/esc50/audio")
files = sorted(A.glob("*.wav"))
meta = [re.match(r"(\d)-(\d+)-([A-Z])-(\d+)\.wav", f.name).groups() for f in files]
clip = np.array([m[1] for m in meta])
y = np.array([int(m[3]) for m in meta])
takes = Counter(clip)
out = {"files": len(files), "source_clips": len(takes), "files_from_multi_take_clips": int(sum(c for c in takes.values() if c > 1))}


def mel_fb(sr, n_fft, n_mels=64):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


FB = None
feats = []
for f in files:
    sr, w = wavfile.read(f)
    w = w.astype(np.float32) / 32768.0
    if FB is None:
        FB = mel_fb(sr, 2048)
    _, _, Z = stft(w, sr, nperseg=2048, noverlap=2048 - 512)
    M = np.log(FB @ (np.abs(Z) ** 2) + 1e-10)
    d = np.diff(M, axis=1)
    feats.append(np.concatenate([M.mean(1), M.std(1), d.mean(1), d.std(1)]))
X = np.array(feats)

# dataset-fingerprint on the flat audio/ folder (as shipped), default grouping vs a group_by regex
work = Path(tempfile.mkdtemp())
os.environ["FINGERPRINT_CACHE"] = str(work / "c.sqlite")


def crossing(v):
    sp = {}
    for r in v.records():
        c = r["source_path"].split("-")[1]
        sp.setdefault(c, set()).add(r["split"])
    multi = [c for c in sp if takes[c] > 1]
    return sum(1 for c in multi if len(sp[c]) > 1), len(multi)


v = fp.Ledger(str(work / "a")).commit(str(A), split="0.8,0.2", quiet=True)
x, m = crossing(v)
v2 = fp.Ledger(str(work / "b")).commit(str(A), split="0.8,0.2", group_by=r"^\d-(\d+)-", quiet=True)
x2, _ = crossing(v2)
out["tool"] = {
    "multi_take_clips": m,
    "clips_split_across_train_test_default_grouping": x,
    "clips_split_across_train_test_group_by_regex": x2,
    "default_group_strategy": v.manifest.get("split_strategy") or {k: v.manifest[k] for k in v.manifest if "strat" in k},
}
shutil.rmtree(work, ignore_errors=True)
print(json.dumps(out), flush=True)


def models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=0),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)),
    }


res = {"random_file": [], "grouped_by_clip": []}
for seed in range(5):
    for k, splitter in (
        ("random_file", ShuffleSplit(1, test_size=0.2, random_state=seed)),
        ("grouped_by_clip", GroupShuffleSplit(1, test_size=0.2, random_state=seed)),
    ):
        tr, te = next(splitter.split(X, groups=clip))
        res[k].append({n: float((mdl.fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()) for n, mdl in models().items()})
    print(seed, res["random_file"][-1], res["grouped_by_clip"][-1], flush=True)
out["accuracy"] = {
    f"{k}/{n}": {"mean": round(float(np.mean([r[n] for r in res[k]])), 4), "std": round(float(np.std([r[n] for r in res[k]])), 4)}
    for k in res
    for n in ("random_forest", "logistic_regression")
}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
