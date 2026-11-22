"""E1b: a memorising model (1-nearest-neighbour on pixels): accuracy on the leaked test images with and
without their duplicates in the training set."""

import csv, json, os, sys, numpy as np

DATA = os.environ.get("BENCH_DATA", ".bench")
B = f"{DATA}/cifar10/bin/"


def load(f):
    r = np.frombuffer(open(B + f, "rb").read(), np.uint8).reshape(-1, 3073)
    return r[:, 1:].astype(np.float32), r[:, 0]


xs, ys = zip(*[load(f"data_batch_{i}.bin") for i in range(1, 6)])
Xtr = np.concatenate(xs)
Ytr = np.concatenate(ys)
Xte, Yte = load("test_batch.bin")
here = os.path.dirname(os.path.abspath(__file__))
cif = [(int(r["TestID"]), int(r["TrainID"])) for r in csv.DictReader(open(f"{DATA}/cifar10/cifair/duplicates_cifar10.csv"))]
ours = [
    (int(r["test"].rsplit("/", 1)[1][:5]), int(r["train"].rsplit("/", 1)[1][:5]))
    for r in csv.DictReader(open(os.path.join(here, "..", "results", "cifar_unlisted_review.csv")))
    if r["review"] == "near-duplicate"
]
lt = np.array(sorted({t for t, _ in cif + ours}))
ltr = np.array(sorted({t for _, t in cif + ours}))


def z(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)


Ztr, Zte = z(Xtr), z(Xte)
del Xtr, Xte


def predict(mask):
    Zm, Ym = Ztr[mask], Ytr[mask]
    return np.concatenate([Ym[(Zte[i : i + 500] @ Zm.T).argmax(1)] for i in range(0, len(Zte), 500)])  # cosine, mean-centred pixels


out = {}
for name, mask in (("full", np.ones(len(Ytr), bool)), ("dedup", ~np.isin(np.arange(len(Ytr)), ltr))):
    ok = predict(mask) == Yte
    clean = np.setdiff1d(np.arange(10000), lt)
    out[name] = {"acc_all": round(float(ok.mean()), 4), "acc_leaked": round(float(ok[lt].mean()), 4), "acc_clean": round(float(ok[clean].mean()), 4)}
print(json.dumps(out))
json.dump(out, open(sys.argv[1], "w"), indent=1)
