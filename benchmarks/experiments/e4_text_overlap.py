"""E4: record-level train/test overlap in public text classification benchmarks, found by
dataset-fingerprint's commit, and what it does to a TF-IDF + logistic-regression model."""

import os

DATA = os.environ.get("BENCH_DATA", ".bench")  # set by download_experiments.sh

import csv, json, shutil, sys, tempfile, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dataset_fingerprint as fp

D = Path(DATA)


def norm(t):
    return " ".join(t.strip().lower().split())


def ag():
    out = {}
    for sp in ("train", "test"):
        rows = list(csv.reader(open(D / f"agnews/data/ag_news_csv/{sp}.csv", encoding="utf-8")))
        out[sp] = [(r[1] + " " + r[2].replace("\\", " "), r[0]) for r in rows]
    return out


def sent(prefix, splits):
    out = {}
    for sp, f in splits.items():
        out[sp] = []
        for line in open(D / "sentconv/data" / f, encoding="latin-1"):
            lab, txt = line.rstrip("\n").split(" ", 1)
            out[sp].append((txt, lab))
    return out


SETS = {
    "AG News": ag,
    "TREC-6": lambda: sent("TREC", {"train": "TREC.train.all", "test": "TREC.test.all"}),
    "SST-2": lambda: sent("SST2", {"train": "stsa.binary.train", "val": "stsa.binary.dev", "test": "stsa.binary.test"}),
    "SST-5": lambda: sent("SST5", {"train": "stsa.fine.train", "val": "stsa.fine.dev", "test": "stsa.fine.test"}),
}

res = []
for name, fn in SETS.items():
    data = fn()
    work = Path(tempfile.mkdtemp())
    for sp, rows in data.items():
        (work / "src" / sp).mkdir(parents=True)
        (work / "src" / sp / "texts.txt").write_text("\n".join(t for t, _ in rows) + "\n", encoding="utf-8")
    os.environ["FINGERPRINT_CACHE"] = str(work / "c.sqlite")
    v = fp.Ledger(str(work / "l")).commit(str(work / "src"), quiet=True)
    lk = v.manifest["quality"]["leakage"]["by_split"]

    # model: does it do better on test texts that also occur in train?
    tr = data["train"]
    train_lab = defaultdict(set)
    for t, l in tr:
        train_lab[norm(t)].add(l)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([t for t, _ in tr])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, [l for _, l in tr])
    ev = {}
    for sp in [s for s in data if s != "train"]:
        rows = data[sp]
        pred = clf.predict(vec.transform([t for t, _ in rows]))
        ok = np.array([p == l for p, (_, l) in zip(pred, rows)])
        leaked = np.array([norm(t) in train_lab for t, _ in rows])
        conflict = sum(1 for (t, l), k in zip(rows, leaked) if k and l not in train_lab[norm(t)])
        ev[sp] = {
            "records": len(rows),
            "records_also_in_train_tool": lk[sp].get("records_also_in_train"),
            "record_leak_rate_tool": lk[sp].get("record_leak_rate"),
            "leaked_records_rowwise": int(leaked.sum()),
            "leaked_with_different_label_in_train": conflict,
            "acc_all": round(float(ok.mean()), 4),
            "acc_leaked": round(float(ok[leaked].mean()), 4) if leaked.any() else None,
            "acc_clean": round(float(ok[~leaked].mean()), 4),
            "acc_if_leaked_removed": round(float(ok[~leaked].mean()), 4),
        }
    r = {"dataset": name, "train_records": len(tr), "splits": ev}
    print(json.dumps(r), flush=True)
    res.append(r)
    shutil.rmtree(work, ignore_errors=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
