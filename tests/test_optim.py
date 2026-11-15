"""Optimisers converge on a simple quadratic; schedulers follow their formulas."""

import math

import numpy as np
import pytest

import neuroforge as nf
from neuroforge import Tensor
from neuroforge.optim import SGD, Adam, AdamW, RMSprop, CosineWarmup, StepLR, OneCycle, clip_grad_norm_


def _minimise(opt_cls, steps=300, **kw):
    target = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
    opt = opt_cls([x], **kw)
    for _ in range(steps):
        loss = ((x - Tensor(target)) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.data, target


@pytest.mark.parametrize("cls,kw", [
    (SGD, dict(lr=0.1)),
    (SGD, dict(lr=0.05, momentum=0.9)),
    (SGD, dict(lr=0.05, momentum=0.9, nesterov=True)),
    (Adam, dict(lr=0.05)),
    (AdamW, dict(lr=0.05, weight_decay=0.0)),
    (RMSprop, dict(lr=0.02)),
])
def test_optimizer_converges(cls, kw):
    x, target = _minimise(cls, **kw)
    assert np.allclose(x, target, atol=1e-2), (cls.__name__, x)


def test_adam_matches_reference_single_step():
    x = Tensor(np.array([1.0]), requires_grad=True)
    opt = Adam([x], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    (x * 3).sum().backward()  # grad = 3
    opt.step()
    # first Adam step moves by exactly lr * sign(grad) (bias-corrected m/v cancel)
    assert np.isclose(x.data[0], 1.0 - 0.1, atol=1e-6)


def test_adamw_decays_only_matrices():
    w = Tensor(np.ones((2, 2)), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    opt = AdamW([w, b], lr=0.1, weight_decay=0.5)
    w.grad, b.grad = np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
    opt.step()
    assert np.allclose(w.data, 1.0 - 0.1 * 0.5)
    assert np.allclose(b.data, 1.0)


def test_clip_grad_norm():
    a = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(np.zeros(4), requires_grad=True)
    a.grad = np.full(3, 3.0, dtype=np.float32)
    b.grad = np.full(4, 4.0, dtype=np.float32)
    total = clip_grad_norm_([a, b], 1.0)
    assert np.isclose(total, math.sqrt(3 * 9 + 4 * 16))
    new_norm = math.sqrt((a.grad ** 2).sum() + (b.grad ** 2).sum())
    assert np.isclose(new_norm, 1.0, atol=1e-4)


def test_cosine_warmup_schedule():
    x = Tensor(np.zeros(1), requires_grad=True)
    opt = Adam([x], lr=1.0)
    sched = CosineWarmup(opt, warmup_steps=10, total_steps=110, min_lr=0.1)
    lrs = [sched.step() for _ in range(120)]
    assert np.isclose(lrs[4], 0.5)              # linear warm-up
    assert np.isclose(lrs[9], 1.0)              # peak at end of warm-up
    assert np.isclose(lrs[59], 0.55, atol=1e-3)  # halfway through cosine
    assert np.isclose(lrs[-1], 0.1)             # floor afterwards
    assert opt.lr == lrs[-1]


def test_step_and_onecycle():
    x = Tensor(np.zeros(1), requires_grad=True)
    opt = Adam([x], lr=1.0)
    s = StepLR(opt, step_size=5, gamma=0.5)
    lrs = [s.step() for _ in range(10)]
    assert lrs[0] == 1.0 and lrs[4] == 0.5 and lrs[9] == 0.25
    opt2 = Adam([x], lr=0.1)
    oc = OneCycle(opt2, total_steps=100, max_lr=1.0)
    lrs = [oc.step() for _ in range(100)]
    assert max(lrs) == pytest.approx(1.0, abs=1e-6)
    assert lrs[-1] < 0.01
