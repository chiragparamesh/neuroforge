"""
Classic 3-arm spiral classification with an MLP. Prints accuracy and an ASCII
decision-boundary map (and saves a PNG if matplotlib is installed).

    python examples/train_mlp_spiral.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuroforge as nf  # noqa: E402
import neuroforge.functional as F  # noqa: E402
from neuroforge import nn  # noqa: E402
from neuroforge.data import make_spiral  # noqa: E402
from neuroforge.optim import Adam  # noqa: E402


def main():
    nf.seed_everything(0)
    X, y = make_spiral(n_per_class=300, n_classes=3, noise=0.15)
    Xt = nf.Tensor(X)
    model = nn.MLP([2, 64, 64, 3], activation=nn.Tanh())
    opt = Adam(model.parameters(), lr=5e-3)

    for epoch in range(1, 1501):
        logits = model(Xt)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 250 == 0:
            acc = (logits.data.argmax(-1) == y).mean()
            print(f"epoch {epoch:4d}  loss {float(loss.data):.4f}  acc {acc * 100:.1f}%")

    # Decision boundary on a grid
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 60), np.linspace(-1.2, 1.2, 30))
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    with nf.no_grad():
        pred = model(nf.Tensor(grid)).data.argmax(-1).reshape(gx.shape)
    chars = ".+#"
    print("\nDecision boundary (ASCII):")
    for row in pred[::-1]:
        print("".join(chars[c] for c in row))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
        grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
        with nf.no_grad():
            pred = model(nf.Tensor(grid)).data.argmax(-1).reshape(gx.shape)
        plt.figure(figsize=(6, 6))
        plt.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.35, cmap="viridis")
        plt.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="viridis")
        plt.title("NeuroForge MLP - spiral decision boundary")
        os.makedirs("checkpoints", exist_ok=True)
        plt.savefig("checkpoints/spiral.png", dpi=120, bbox_inches="tight")
        print("saved checkpoints/spiral.png")
    except ImportError:
        print("(install matplotlib to save a PNG of the decision boundary)")


if __name__ == "__main__":
    main()
