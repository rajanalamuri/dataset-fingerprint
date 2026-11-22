"""Second, automatic rater for the flagged CIFAR pairs: shift-tolerant normalised cross-correlation of
grayscale pixels (invariant to brightness, contrast and grayscale conversion). Calibrated on ciFAIR's
own labelled pairs and on random train/test pairs."""

import os

DATA = os.environ.get("BENCH_DATA", ".bench")  # set by download_experiments.sh

import csv, json, numpy as np

B = f"{DATA}/cifar10/bin/"


def load(f):
    r = np.frombuffer(open(B + f, "rb").read(), np.uint8).reshape(-1, 3073)
    return r[:, 1:].reshape(-1, 3, 32, 32)


tr = np.concatenate([load(f"data_batch_{i}.bin") for i in range(1, 6)])
te = load("test_batch.bin")
g = lambda x: (x.astype(np.float32) * np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)).sum(0)


def ncc(a, b, s=2):
    A, Bm = g(a), g(b)
    best = -1
    for dx in range(-s, s + 1):
        for dy in range(-s, s + 1):
            x = A[max(0, dy) : 32 + min(0, dy), max(0, dx) : 32 + min(0, dx)]
            y = Bm[max(0, -dy) : 32 + min(0, -dy), max(0, -dx) : 32 + min(0, -dx)]
            for yy in (y, y[:, ::-1]) if False else (y,):
                x0 = (x - x.mean()) / (x.std() + 1e-6)
                y0 = (yy - yy.mean()) / (yy.std() + 1e-6)
                best = max(best, float((x0 * y0).mean()))
    return best


cif = list(csv.DictReader(open(f"{DATA}/cifar10/cifair/duplicates_cifar10.csv")))
cal = {
    "cifair_near_duplicate": [ncc(te[int(r["TestID"])], tr[int(r["TrainID"])]) for r in cif if r["Judgment"] == "1"],
    "cifair_very_similar": [ncc(te[int(r["TestID"])], tr[int(r["TrainID"])]) for r in cif if r["Judgment"] == "2"],
}
rng = np.random.default_rng(0)
cal["random_pairs"] = [ncc(te[a], tr[b]) for a, b in zip(rng.integers(0, 10000, 500), rng.integers(0, 50000, 500))]
rev = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "results", "cifar_unlisted_review.csv"))))
idx = lambda p: int(p.rsplit("/", 1)[1][:5])
for r in rev:
    r["ncc"] = round(ncc(te[idx(r["test"])], tr[idx(r["train"])]), 3)
thr = float(np.percentile(cal["cifair_near_duplicate"], 5))
summ = {
    k: {
        "n": len(v),
        "p5": round(float(np.percentile(v, 5)), 3),
        "median": round(float(np.median(v)), 3),
        "p95": round(float(np.percentile(v, 95)), 3),
    }
    for k, v in cal.items()
}
summ["threshold_p5_of_cifair_near_duplicates"] = round(thr, 3)
summ["random_pairs_above_threshold"] = int(sum(x >= thr for x in cal["random_pairs"]))
agree = [(r["review"], r["ncc"] >= thr) for r in rev]
summ["review_vs_rater2"] = {
    f"{a}|{'dup' if b else 'not'}": sum(1 for x in agree if x == (a, b)) for a in ("near-duplicate", "different") for b in (True, False)
}
summ["pairs"] = [{k: r[k] for k in ("test", "train", "dhash_distance", "review", "ncc")} for r in rev]
print(json.dumps({k: v for k, v in summ.items() if k != "pairs"}, indent=1))
json.dump(summ, open("rater2.json", "w"), indent=1)
