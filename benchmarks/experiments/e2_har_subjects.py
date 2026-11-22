"""E2: subject/recording leakage in sensor data (UCI HAR, 30 subjects, 10,299 windows of 2.56 s with
50 % overlap). Compares a random window split (what a generic splitter does) with a subject split, and
checks what dataset-fingerprint does with each layout."""

import os

DATA = os.environ.get("BENCH_DATA", ".bench")  # set by download_experiments.sh

import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dataset_fingerprint as fp

H = Path(f"{DATA}/har")
X = np.vstack([np.loadtxt(H / "train/X_train.txt"), np.loadtxt(H / "test/X_test.txt")])
y = np.concatenate([np.loadtxt(H / "train/y_train.txt"), np.loadtxt(H / "test/y_test.txt")]).astype(int)
subj = np.concatenate([np.loadtxt(H / "train/subject_train.txt"), np.loadtxt(H / "test/subject_test.txt")]).astype(int)
sig = []
for sp in ("train", "test"):
    chans = sorted((H / sp / "Inertial Signals").glob("*.txt"))
    sig.append(np.stack([np.loadtxt(c) for c in chans], axis=1).astype(np.float32))  # (n, 9, 128)
S = np.concatenate(sig)
n = len(y)
print("windows", n, "subjects", len(set(subj)), flush=True)

out = {"windows": int(n), "subjects": int(len(set(subj)))}
work = Path(tempfile.mkdtemp())
os.environ["FINGERPRINT_CACHE"] = str(work / "c.sqlite")


def write(root, split_of):
    for i in range(n):
        d = root / (split_of[i] if split_of is not None else "") / f"subject_{subj[i]:02d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"win_{i:05d}.npy", S[i])


# (a) one folder per subject, let dataset-fingerprint split it
write(work / "flat", None)
v = fp.Ledger(str(work / "la")).commit(str(work / "flat"), split="0.7,0.3", quiet=True)
recs = v.records()
sp_of = {int(r["source_path"].rsplit("_", 1)[1][:5]): r["split"] for r in recs}
subj_splits = {}
for i, s in sp_of.items():
    subj_splits.setdefault(int(subj[i]), set()).add(s)
out["tool_split"] = {
    "strategy": v.manifest.get("split", v.manifest.get("splits_strategy")),
    "groups": len({r["source_path"].split("/")[0] for r in recs}),
    "subjects_in_more_than_one_split": sum(1 for v_ in subj_splits.values() if len(v_) > 1),
    "files_by_split": {k: sum(1 for x in sp_of.values() if x == k) for k in sorted(set(sp_of.values()))},
}
tool_test = np.array([sp_of[i] != "train" for i in range(n)])

# (b) a random window split handed to the tool as train/ test/ folders
rs = ShuffleSplit(1, test_size=0.3, random_state=0)
tr_idx, te_idx = next(rs.split(X))
split_of = np.array(["train"] * n, dtype=object)
split_of[te_idx] = "test"
write(work / "random", split_of)
v2 = fp.Ledger(str(work / "lb")).commit(str(work / "random"), quiet=True)
lk = v2.manifest["quality"]["leakage"]
out["tool_on_random_split"] = {
    "leak_rate": lk["leak_rate"],
    "recording_names_also_in_train": lk.get("recording_names_also_in_train", {}).get("count"),
    "warning": next((w for w in v2.manifest["lineage"]["warnings"] if "recording" in w), None),
}
shutil.rmtree(work, ignore_errors=True)
print(json.dumps(out), flush=True)


def models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    }


def score(tr, te):
    return {k: float((m.fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()) for k, m in models().items()}


acc = {"random_window": [], "subject": []}
for seed in range(5):
    tr, te = next(ShuffleSplit(1, test_size=0.3, random_state=seed).split(X))
    acc["random_window"].append(score(tr, te))
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, groups=subj))
    acc["subject"].append(score(tr, te))
    print(seed, acc["random_window"][-1], acc["subject"][-1], flush=True)
acc["dataset_fingerprint_default"] = score(np.where(~tool_test)[0], np.where(tool_test)[0])
summ = {}
for k in ("random_window", "subject"):
    for mname in ("random_forest", "logistic_regression"):
        vals = [a[mname] for a in acc[k]]
        summ[f"{k}/{mname}"] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
out["accuracy"] = summ
out["accuracy_tool_split"] = {k: round(v, 4) for k, v in acc["dataset_fingerprint_default"].items()}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
