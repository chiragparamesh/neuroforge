"""
Train a convolutional network on a synthetic shapes dataset (circle / square / triangle).

Everything - the conv layers, im2col, max-pooling and their backward passes -
is implemented in NeuroForge, so this verifies the whole CNN stack end to end.

    python examples/train_cnn_shapes.py --epochs 5 --dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuroforge as nf  # noqa: E402
import neuroforge.functional as F  # noqa: E402
from neuroforge import nn  # noqa: E402
from neuroforge.data import DataLoader, make_shapes  # noqa: E402
from neuroforge.optim import Adam  # noqa: E402
from neuroforge.utils import count_parameters, save_checkpoint  # noqa: E402


class ShapeNet(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(model, X, y, batch=256):
    model.eval()
    correct = 0
    with nf.no_grad():
        for i in range(0, len(X), batch):
            logits = model(nf.Tensor(X[i: i + batch]))
            correct += int((logits.data.argmax(-1) == y[i: i + batch]).sum())
    model.train()
    return correct / len(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--out", default="checkpoints/shapes_cnn.npz")
    args = p.parse_args()
    nf.seed_everything(0)

    X, y = make_shapes(n=args.n, seed=0)
    n_train = int(0.8 * len(X))
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    loader = DataLoader(Xtr, ytr, batch_size=args.batch)
    model = ShapeNet()
    opt = Adam(model.parameters(), lr=args.lr)
    print(f"ShapeNet: {count_parameters(model)} params | train {n_train} test {len(Xte)}\n{model}")

    dash = None
    if args.dashboard:
        from neuroforge.dashboard import Dashboard
        dash = Dashboard(port=args.port, title="ShapeNet CNN", open_browser=True)
        dash.set_config(params=count_parameters(model), epochs=args.epochs, batch=args.batch)

    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        for xb, yb in loader:
            logits = model(nf.Tensor(xb))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if dash:
                dash.log(step, loss=float(loss.data), lr=opt.lr)
            if step % 10 == 0:
                print(f"\repoch {epoch} step {step} loss {float(loss.data):.4f}", end="")
        acc = accuracy(model, Xte, yte)
        print(f"\nepoch {epoch}: test accuracy {acc * 100:.2f}%  ({time.time() - t0:.0f}s)")
        if dash:
            dash.log(step, accuracy=acc)
    save_checkpoint(args.out, model, extra={"accuracy": acc})
    print(f"saved {args.out}")
    if dash:
        dash.finish()


if __name__ == "__main__":
    main()
