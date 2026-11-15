"""
neuroforge.data.dataset
=======================

Tiny data utilities: a language-model batch sampler and a generic
``DataLoader`` for array datasets.
"""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np


class TextDataset:
    """Holds a token id stream and samples ``(x, y)`` next-token-prediction batches."""

    def __init__(self, ids: Sequence[int], block_size: int, split: float = 0.9):
        ids = np.asarray(ids, dtype=np.int64)
        n = int(len(ids) * split)
        self.train_ids, self.val_ids = ids[:n], ids[n:]
        self.block_size = block_size

    def get_batch(self, batch_size: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        data = self.train_ids if split == "train" else self.val_ids
        ix = np.random.randint(0, len(data) - self.block_size - 1, size=batch_size)
        x = np.stack([data[i: i + self.block_size] for i in ix])
        y = np.stack([data[i + 1: i + 1 + self.block_size] for i in ix])
        return x, y


class DataLoader:
    """Iterate over ``(X, y)`` arrays in shuffled mini-batches."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        assert len(X) == len(y)
        self.X, self.y, self.batch_size, self.shuffle = X, y, batch_size, shuffle

    def __len__(self) -> int:
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.random.permutation(len(self.X)) if self.shuffle else np.arange(len(self.X))
        for i in range(0, len(idx), self.batch_size):
            b = idx[i: i + self.batch_size]
            yield self.X[b], self.y[b]


# --------------------------------------------------------------------------- #
# Synthetic datasets (no downloads needed)
# --------------------------------------------------------------------------- #
def make_spiral(n_per_class: int = 200, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """The classic multi-arm spiral classification problem."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_per_class * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        ix = range(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.randn(n_per_class) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = c
    return X, y


def make_shapes(n: int = 2000, size: int = 28, seed: int = 0):
    """Synthetic image dataset: circles (0), squares (1), triangles (2) at random positions and sizes."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = rng.randint(0, 3, size=n).astype(np.int64)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        cx, cy = rng.uniform(size * 0.3, size * 0.7, size=2)
        r = rng.uniform(size * 0.15, size * 0.3)
        if y[i] == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        elif y[i] == 1:
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        else:
            mask = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= (yy - (cy - r)) / 2)
        img = mask.astype(np.float32)
        img += rng.randn(size, size).astype(np.float32) * 0.1
        X[i, 0] = np.clip(img, 0, 1)
    return X, y


def make_arithmetic(n: int = 20000, n_digits: int = 3, seed: int = 0):
    """Strings like ``"123+456=579;"`` for teaching a GPT to add numbers.

    The answer is written reversed-free (plain) and padded so every sample has
    the same length, which makes batching trivial.
    """
    rng = np.random.RandomState(seed)
    hi = 10 ** n_digits
    lines = []
    for _ in range(n):
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        lines.append(f"{a:0{n_digits}d}+{b:0{n_digits}d}={a + b:0{n_digits + 1}d};")
    return lines
