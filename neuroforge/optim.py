"""
neuroforge.optim
================

Optimisers and learning-rate schedulers.

All optimisers operate in-place on ``Tensor.data`` using the ``.grad`` that
``backward()`` accumulated. Call ``zero_grad()`` before each backward pass.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np

from .tensor import Tensor


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params: List[Tensor] = list(params)
        assert len(self.params) > 0, "optimizer got an empty parameter list"
        self.lr = lr
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"lr": self.lr, "t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.lr = state.get("lr", self.lr)
        self.t = state.get("t", self.t)


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum, Nesterov and weight decay."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0, nesterov: bool = False,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params] if momentum > 0 else None

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.velocity is not None:
                v = self.velocity[i]
                v *= self.momentum
                v += g
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data           # L2 (coupled) weight decay
            m, v = self.m[i], self.v[i]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


class AdamW(Adam):
    """Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019). Used by GPT-2/3, LLaMA, etc."""

    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        super().__init__(params, lr, betas, eps, weight_decay=0.0)
        self.decoupled_wd = weight_decay

    def step(self) -> None:
        if self.decoupled_wd:
            for p in self.params:
                if p.grad is not None and p.data.ndim >= 2:  # do not decay biases / norms
                    p.data -= self.lr * self.decoupled_wd * p.data
        super().step()


class RMSprop(Optimizer):
    def __init__(self, params, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8):
        super().__init__(params, lr)
        self.alpha, self.eps = alpha, eps
        self.sq = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            s = self.sq[i]
            s *= self.alpha
            s += (1 - self.alpha) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(s) + self.eps)


# --------------------------------------------------------------------------- #
# Gradient utilities
# --------------------------------------------------------------------------- #
def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is at most ``max_norm``. Returns the norm."""
    params = [p for p in params if p.grad is not None]
    total = math.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params))
    if total > max_norm and total > 0:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale
    return total


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #
class LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def get_lr(self, step: int) -> float:
        raise NotImplementedError

    def step(self) -> float:
        self.step_count += 1
        lr = self.get_lr(self.step_count)
        self.optimizer.lr = lr
        return lr


class CosineWarmup(LRScheduler):
    """Linear warm-up followed by cosine decay to ``min_lr`` (the GPT recipe)."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: Optional[float] = None):
        super().__init__(optimizer)
        self.warmup, self.total = warmup_steps, total_steps
        self.min_lr = self.base_lr * 0.1 if min_lr is None else min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * step / max(1, self.warmup)
        if step >= self.total:
            return self.min_lr
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (self.base_lr - self.min_lr)


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self, step: int) -> float:
        return self.base_lr * (self.gamma ** (step // self.step_size))


class OneCycle(LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, max_lr: Optional[float] = None, pct_start: float = 0.3):
        super().__init__(optimizer)
        self.total = total_steps
        self.max_lr = max_lr or self.base_lr * 10
        self.pct_start = pct_start

    def get_lr(self, step: int) -> float:
        up = int(self.total * self.pct_start)
        if step <= up:
            return self.base_lr + (self.max_lr - self.base_lr) * step / max(1, up)
        progress = (step - up) / max(1, self.total - up)
        return self.base_lr * 0.01 + 0.5 * (1 + math.cos(math.pi * progress)) * (self.max_lr - self.base_lr * 0.01)
