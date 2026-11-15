"""
neuroforge.utils
================

Gradient checking, checkpointing, seeding and a tiny progress bar.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np

from .tensor import Tensor, default_dtype


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def gradcheck(fn: Callable[..., Tensor], *inputs: Tensor, eps: float = 1e-3, atol: float = 1e-2,
              rtol: float = 1e-2, verbose: bool = False) -> bool:
    """Compare analytic gradients from ``backward()`` with central finite differences.

    ``fn`` must return a scalar Tensor. Inputs are evaluated in float64 for accuracy.
    """
    # Promote to float64 copies so finite differences are meaningful.
    with default_dtype(np.float64):
        return _gradcheck64(fn, inputs, eps, atol, rtol, verbose)


def _gradcheck64(fn, inputs, eps, atol, rtol, verbose) -> bool:
    inputs64 = [Tensor(t.data.astype(np.float64), requires_grad=True, dtype=np.float64) for t in inputs]
    out = fn(*inputs64)
    out.backward()
    ok = True
    for k, x in enumerate(inputs64):
        analytic = x.grad.astype(np.float64)
        numeric = np.zeros_like(x.data, dtype=np.float64)
        it = np.nditer(x.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = x.data[idx]
            x.data[idx] = orig + eps
            fp = float(fn(*inputs64).data)
            x.data[idx] = orig - eps
            fm = float(fn(*inputs64).data)
            x.data[idx] = orig
            numeric[idx] = (fp - fm) / (2 * eps)
            it.iternext()
        err = np.abs(analytic - numeric)
        tol = atol + rtol * np.abs(numeric)
        bad = err > tol
        if bad.any():
            ok = False
            if verbose:
                print(f"[gradcheck] input {k}: max abs err {err.max():.3e} at {np.unravel_index(err.argmax(), err.shape)}")
                print("  analytic:", analytic[bad][:5], "\n  numeric: ", numeric[bad][:5])
        elif verbose:
            print(f"[gradcheck] input {k}: OK (max abs err {err.max():.3e})")
    return ok


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer=None, extra: Optional[Dict] = None) -> None:
    """Save a model (and optional optimiser/extra metadata) to a ``.npz`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {f"model/{k}": v for k, v in model.state_dict().items()}
    meta = {"extra": extra or {}}
    if hasattr(model, "cfg") and hasattr(model.cfg, "to_dict"):
        meta["config"] = model.cfg.to_dict()
    if optimizer is not None:
        meta["optimizer"] = optimizer.state_dict()
    np.savez(path, __meta__=json.dumps(meta), **arrays)


def load_checkpoint(path: str, model=None, optimizer=None) -> Dict:
    """Load a checkpoint. Returns the metadata dict; loads weights into ``model`` if given."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        state = {k[len("model/"):]: z[k] for k in z.files if k.startswith("model/")}
    if model is not None:
        model.load_state_dict(state)
    if optimizer is not None and "optimizer" in meta:
        optimizer.load_state_dict(meta["optimizer"])
    meta["state_dict"] = state
    return meta


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
class Progress:
    """Minimal in-place progress line, e.g. ``step 120/1000  loss 1.234  lr 3.0e-04  4.2 it/s``."""

    def __init__(self, total: int, every: int = 1):
        self.total, self.every = total, every
        self.start = time.time()

    def update(self, step: int, **metrics) -> None:
        if step % self.every and step != self.total:
            return
        elapsed = time.time() - self.start
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (self.total - step) / rate if rate > 0 else 0.0
        parts = [f"step {step}/{self.total}"]
        for k, v in metrics.items():
            parts.append(f"{k} {v:.4f}" if isinstance(v, float) and abs(v) >= 1e-2 else f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}")
        parts.append(f"{rate:.1f} it/s  eta {eta:5.0f}s")
        sys.stdout.write("\r" + "  ".join(parts) + " " * 8)
        sys.stdout.flush()
        if step == self.total:
            sys.stdout.write("\n")


def count_parameters(model) -> str:
    n = model.num_parameters()
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
