"""E1: does a train/test near-duplicate inflate test accuracy on that image?
Same test images, two training sets: (A) full CIFAR-10 train, (B) train minus the matched duplicates.
Accuracy on the leaked test images under A vs B is the inflation caused by the leak."""

import os

DATA = os.environ.get("BENCH_DATA", ".bench")  # set by download_experiments.sh

import csv, json, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)
B = f"{DATA}/cifar10/bin/"


def load(f):
    r = np.frombuffer(open(B + f, "rb").read(), np.uint8).reshape(-1, 3073)
    return r[:, 1:].reshape(-1, 3, 32, 32), r[:, 0].astype(np.int64)


xs, ys = zip(*[load(f"data_batch_{i}.bin") for i in range(1, 6)])
Xtr = np.concatenate(xs)
Ytr = np.concatenate(ys)
Xte, Yte = load("test_batch.bin")
cif = [(int(r["TestID"]), int(r["TrainID"])) for r in csv.DictReader(open(f"{DATA}/cifar10/cifair/duplicates_cifar10.csv"))]
ours = []
for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "results", "cifar_unlisted_review.csv"))):
    if r["review"] == "near-duplicate":
        ours.append((int(r["test"].rsplit("/", 1)[1][:5]), int(r["train"].rsplit("/", 1)[1][:5])))
pairs = cif + ours
leak_test = sorted({t for t, _ in pairs})
leak_train = sorted({t for _, t in pairs})
mean = torch.tensor([125.3, 123.0, 113.9]).view(1, 3, 1, 1)
std = torch.tensor([63.0, 62.1, 66.7]).view(1, 3, 1, 1)


def net():
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

    return nn.Sequential(
        blk(3, 32),
        nn.MaxPool2d(2),
        blk(32, 64),
        nn.MaxPool2d(2),
        blk(64, 128),
        nn.MaxPool2d(2),
        blk(128, 256),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10),
    )


def prep(x):
    return (torch.from_numpy(x).float() - mean) / std


def train(X, Y, seed, epochs):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = net()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    n = len(X)
    steps = epochs * ((n + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    Xt = prep(X)
    Yt = torch.from_numpy(Y)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            xb = Xt[idx]
            if True:  # flip + random crop (pad 4)
                flip = torch.rand(len(idx), generator=g) < 0.5
                xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
                xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
                dx, dy = torch.randint(0, 9, (2,), generator=g)
                xb = xb[:, :, dy : dy + 32, dx : dx + 32]
            loss = F.cross_entropy(m(xb), Yt[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    preds = []
    with torch.no_grad():
        Xe = prep(Xte)
        for i in range(0, len(Xe), 500):
            preds.append(m(Xe[i : i + 500]).argmax(1))
    return torch.cat(preds).numpy()


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2].split(",")]
    out = sys.argv[3]
    keep = np.setdiff1d(np.arange(len(Xtr)), leak_train)
    res = []
    for seed in seeds:
        for cond, (X, Y) in (("full", (Xtr, Ytr)), ("dedup", (Xtr[keep], Ytr[keep]))):
            t = time.time()
            p = train(X, Y, seed, epochs)
            ok = p == Yte
            lt = np.array(leak_test)
            clean = np.setdiff1d(np.arange(10000), lt)
            r = {
                "seed": seed,
                "condition": cond,
                "train_size": len(X),
                "epochs": epochs,
                "acc_all": float(ok.mean()),
                "acc_leaked": float(ok[lt].mean()),
                "acc_clean": float(ok[clean].mean()),
                "acc_cifair": float(ok[[t for t, _ in cif]].mean()),
                "acc_ours24": float(ok[sorted({t for t, _ in ours})].mean()),
                "n_leaked": len(lt),
                "seconds": round(time.time() - t),
            }
            print(json.dumps(r), flush=True)
            res.append(r)
            json.dump(res, open(out, "w"), indent=1)
