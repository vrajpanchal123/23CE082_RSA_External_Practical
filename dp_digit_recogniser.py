"""
Responsible and Safe AI (AIML4001) - External Practical, Set D (25/11/2026)
Privacy-Preserving Training of a PIN-Code Digit Recogniser (DP-SGD with Opacus)

Dataset : MNIST (20,000-image training subset, full 10,000-image test set)
Settings: seed 42, GroupNorm CNN, delta=1e-5, max_grad_norm=1.0, batch 256, 3 epochs, CPU
Tasks   : 1 baseline | 2 DP eps=1,5,10 | 3 fixed noise 1.1 (eps/epoch)
          4 membership-inference confidence check | 5 plots | 6 results table (JSON)
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

# ----------------------------- configuration -----------------------------
SEED = 42
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 256
EPOCHS = 3
SUBSET = 20000
LR = 0.1                       # SGD learning rate (same for all runs)
FIXED_NOISE = 1.1
TARGET_EPS = [10.0, 5.0, 1.0]
DATA_DIR = os.environ.get("MNIST_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", ".")
DEVICE = torch.device("cpu")
torch.set_num_threads(2)


# ------------------------------- data ------------------------------------
def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_full), generator=g)[:SUBSET].tolist()
    train = Subset(train_full, idx)
    return train, test


def loaders(train, test):
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    train_eval = DataLoader(train, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False)
    return train_loader, train_eval, test_loader


# ------------------------------- model -----------------------------------
class CNN(nn.Module):
    """Small CNN; GroupNorm replaces BatchNorm (BatchNorm is not Opacus-compatible)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)   # 14x14
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)   # 7x7
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


# --------------------------- train / evaluate -----------------------------
def train_epoch(model, loader, opt):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


@torch.no_grad()
def evaluate(model, loader):
    """Returns (accuracy %, mean max-softmax confidence)."""
    model.eval()
    correct, conf, n = 0, 0.0, 0
    for x, y in loader:
        p = F.softmax(model(x.to(DEVICE)), dim=1)
        correct += (p.argmax(1) == y).sum().item()
        conf += p.max(1).values.sum().item()
        n += len(y)
    return 100.0 * correct / n, conf / n


def run_baseline(train, test):
    torch.manual_seed(SEED)
    tr, tr_eval, te = loaders(train, test)
    model = CNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, tr, opt)
        print(f"  [baseline] epoch {ep} test acc {evaluate(model, te)[0]:.2f}%", flush=True)
    acc, te_conf = evaluate(model, te)
    _, tr_conf = evaluate(model, tr_eval)
    return dict(acc=acc, train_conf=tr_conf, test_conf=te_conf)


def run_dp(train, test, target_eps=None, noise=None):
    torch.manual_seed(SEED)
    tr, tr_eval, te = loaders(train, test)
    model = CNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    pe = PrivacyEngine()
    if target_eps is not None:
        model, opt, tr = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=tr, epochs=EPOCHS,
            target_epsilon=target_eps, target_delta=DELTA, max_grad_norm=MAX_GRAD_NORM)
        tag = f"eps={target_eps:g}"
    else:
        model, opt, tr = pe.make_private(
            module=model, optimizer=opt, data_loader=tr,
            noise_multiplier=noise, max_grad_norm=MAX_GRAD_NORM)
        tag = f"sigma={noise}"
    eps_per_epoch = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, tr, opt)
        eps_per_epoch.append(pe.get_epsilon(DELTA))
        print(f"  [DP {tag}] epoch {ep} test acc {evaluate(model, te)[0]:.2f}% "
              f"eps {eps_per_epoch[-1]:.2f}", flush=True)
    acc, te_conf = evaluate(model, te)
    _, tr_conf = evaluate(model, tr_eval)
    return dict(acc=acc, train_conf=tr_conf, test_conf=te_conf,
                eps_spent=eps_per_epoch[-1], eps_per_epoch=eps_per_epoch,
                noise_multiplier=float(opt.noise_multiplier))


# --------------------------------- plots ----------------------------------
def make_plots(base, dp, fixed):
    eps = sorted(dp)  # ascending
    accs = [dp[e]["acc"] for e in eps]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(eps, accs, "o-", color="tab:blue", lw=2, label="DP-SGD (Opacus)")
    for e, a in zip(eps, accs):
        plt.annotate(f"{a:.1f}%", (e, a), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.axhline(base["acc"], color="tab:red", ls="--", label=f"Non-private baseline ({base['acc']:.1f}%)")
    plt.xlabel("Privacy budget  ε  (δ = 1e-5)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Privacy-utility trade-off (MNIST, 3 epochs)")
    plt.xticks(eps)
    plt.grid(alpha=.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_epsilon.png"), dpi=150); plt.close()

    plt.figure(figsize=(6.5, 4.5))
    ep = list(range(1, EPOCHS + 1))
    plt.plot(ep, fixed["eps_per_epoch"], "s-", color="tab:green", lw=2)
    for x, y in zip(ep, fixed["eps_per_epoch"]):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("Epoch"); plt.ylabel("ε spent (δ = 1e-5)")
    plt.title("Privacy spent per epoch (noise multiplier = 1.1)")
    plt.xticks(ep); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "epsilon_vs_epoch.png"), dpi=150); plt.close()


# --------------------------------- main -----------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test = get_data()
    print(f"Train subset: {len(train)} | Test: {len(test)} | device: {DEVICE}")

    print("\n== Task 1: non-private baseline ==")
    base = run_baseline(train, test)
    print(f"Baseline (no DP)   : test accuracy {base['acc']:.1f} %")

    print("\n== Task 2: DP models for target eps = 10, 5, 1 ==")
    dp = {e: run_dp(train, test, target_eps=e) for e in TARGET_EPS}
    print("\nTARGET_EPS   DELTA    CLIP(C)   EPOCHS   TEST_ACC   EPS_SPENT")
    print("-" * 61)
    for e in TARGET_EPS:
        r = dp[e]
        print(f"{e:7.1f}      {DELTA:.0e}      {MAX_GRAD_NORM:.1f}        {EPOCHS}      "
              f"{r['acc']:5.1f} %      {r['eps_spent']:.2f}")
    print("-" * 61)
    cost = dp[1.0]["acc"] - base["acc"]
    print(f"Accuracy cost of strongest privacy (eps = 1) : {cost:+.1f} %")

    print("\n== Task 3: fixed noise_multiplier = 1.1 ==")
    fixed = run_dp(train, test, noise=FIXED_NOISE)
    print("EPOCH    EPSILON_SPENT\n" + "-" * 21)
    for i, e in enumerate(fixed["eps_per_epoch"], 1):
        print(f"{i:3d}        {e:6.2f}")
    print("-" * 21)

    print("\n== Task 4: membership-inference check (mean max-softmax) ==")
    rows = [("Baseline", base), ("DP (eps=1)", dp[1.0])]
    print("MODEL         TRAIN    TEST    GAP\n" + "-" * 35)
    for name, r in rows:
        print(f"{name:<12}  {r['train_conf']:.3f}    {r['test_conf']:.3f}   "
              f"{r['train_conf'] - r['test_conf']:.3f}")
    print("-" * 35)

    print("\n== Task 5: plots ==")
    make_plots(base, dp, fixed)
    print("Plot saved : accuracy_vs_epsilon.png, epsilon_vs_epoch.png")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(dict(baseline=base, dp={str(k): v for k, v in dp.items()},
                       fixed_noise=fixed, config=dict(
                           seed=SEED, delta=DELTA, clip=MAX_GRAD_NORM, batch=BATCH_SIZE,
                           epochs=EPOCHS, subset=SUBSET, lr=LR)), f, indent=2)
    print("Results saved : results.json")


if __name__ == "__main__":
    main()
