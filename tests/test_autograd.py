"""Gradient correctness of every Tensor op, checked against central finite differences."""

import numpy as np
import pytest

import neuroforge as nf
from neuroforge import Tensor
from neuroforge.utils import gradcheck


def rnd(*shape):
    return Tensor(np.random.randn(*shape))


W34 = Tensor(np.random.RandomState(7).randn(3, 4))


@pytest.mark.parametrize("name,fn,shapes", [
    ("add", lambda a, b: (a + b).sum(), [(3, 4), (3, 4)]),
    ("add_broadcast", lambda a, b: (a + b).sum(), [(3, 4), (4,)]),
    ("sub", lambda a, b: ((a - b) ** 2).sum(), [(3, 4), (1, 4)]),
    ("mul", lambda a, b: (a * b).sum(), [(3, 4), (3, 1)]),
    ("div", lambda a, b: (a / (b * b + 1.0)).sum(), [(3, 4), (3, 4)]),
    ("pow", lambda a: (a ** 3).sum(), [(3, 4)]),
    ("matmul", lambda a, b: (a @ b).sum(), [(3, 4), (4, 5)]),
    ("matmul_batched", lambda a, b: (a @ b).sum(), [(2, 3, 4), (2, 4, 5)]),
    ("matmul_broadcast", lambda a, b: (a @ b).sum(), [(2, 3, 4), (4, 5)]),
    ("matvec", lambda a, b: (a @ b).sum(), [(3, 4), (4,)]),
    ("exp", lambda a: a.exp().sum(), [(3, 4)]),
    ("log", lambda a: (a * a + 0.5).log().sum(), [(3, 4)]),
    ("sqrt", lambda a: (a * a + 1.0).sqrt().sum(), [(3, 4)]),
    ("abs", lambda a: (a.abs() * W34).sum(), [(3, 4)]),
    ("relu", lambda a: (a.relu() * W34).sum(), [(3, 4)]),
    ("leaky_relu", lambda a: (a.leaky_relu(0.1) * W34).sum(), [(3, 4)]),
    ("sigmoid", lambda a: (a.sigmoid() * W34).sum(), [(3, 4)]),
    ("tanh", lambda a: (a.tanh() * W34).sum(), [(3, 4)]),
    ("gelu", lambda a: (a.gelu() * W34).sum(), [(3, 4)]),
    ("silu", lambda a: (a.silu() * W34).sum(), [(3, 4)]),
    ("softmax", lambda a: (a.softmax(-1) * W34).sum(), [(3, 4)]),
    ("softmax_axis0", lambda a: (a.softmax(0) * W34).sum(), [(3, 4)]),
    ("log_softmax", lambda a: (a.log_softmax(-1) * W34).sum(), [(3, 4)]),
    ("clip", lambda a: (a.clip(-0.5, 0.5) * W34).sum(), [(3, 4)]),
    ("sum_axis", lambda a: (a.sum(axis=1) * Tensor([1.0, 2.0, 3.0])).sum(), [(3, 4)]),
    ("sum_keepdims", lambda a: (a.sum(axis=0, keepdims=True) * Tensor([[1.0, 2.0, 3.0, 4.0]])).sum(), [(3, 4)]),
    ("mean", lambda a: (a.mean(axis=(0, 1))), [(3, 4)]),
    ("var", lambda a: (a.var(axis=1) * Tensor([1.0, 2.0, 3.0])).sum(), [(3, 4)]),
    ("std", lambda a: (a.std(axis=0) * Tensor([1.0, 2.0, 3.0, 4.0])).sum(), [(3, 4)]),
    ("max", lambda a: (a.max(axis=1) * Tensor([1.0, 2.0, 3.0])).sum(), [(3, 4)]),
    ("min", lambda a: (a.min(axis=0) * Tensor([1.0, 2.0, 3.0, 4.0])).sum(), [(3, 4)]),
    ("max_all", lambda a: a.max(), [(3, 4)]),
    ("reshape", lambda a: (a.reshape(4, 3) * W34.T).sum(), [(3, 4)]),
    ("transpose", lambda a: (a.T @ a).sum(), [(3, 4)]),
    ("permute", lambda a: (a.permute(2, 0, 1) ** 2 * Tensor(np.arange(24.).reshape(4, 2, 3))).sum(), [(2, 3, 4)]),
    ("transpose_swap", lambda a: (a.transpose(1, 2) * Tensor(np.arange(24.).reshape(2, 4, 3))).sum(), [(2, 3, 4)]),
    ("expand", lambda a: (a.expand(3, 4) * W34).sum(), [(1, 4)]),
    ("unsqueeze_squeeze", lambda a: (a.unsqueeze(1).squeeze(1) * W34).sum(), [(3, 4)]),
    ("index_slice", lambda a: (a[1:, ::2] ** 2).sum(), [(3, 4)]),
    ("index_fancy", lambda a: (a[np.array([0, 0, 2]), np.array([1, 1, 3])] * Tensor([1.0, 2.0, 3.0])).sum(), [(3, 4)]),
    ("masked_fill", lambda a: (a.masked_fill(np.eye(3, 4, dtype=bool), -1.0) * W34).sum(), [(3, 4)]),
    ("pad", lambda a: (a.pad(((1, 1), (0, 2))) ** 2 * Tensor(np.arange(30.).reshape(5, 6))).sum(), [(3, 4)]),
    ("cat", lambda a, b: (nf.cat([a, b], axis=0) * Tensor(np.arange(24.).reshape(6, 4))).sum(), [(3, 4), (3, 4)]),
    ("stack", lambda a, b: (nf.stack([a, b], axis=1) * Tensor(np.arange(24.).reshape(3, 2, 4))).sum(), [(3, 4), (3, 4)]),
    ("where", lambda a, b: (nf.where(np.eye(3, 4, dtype=bool), a, b) * W34).sum(), [(3, 4), (3, 4)]),
    ("chain", lambda a, b: ((a @ b).tanh().softmax(-1).log() * Tensor(np.random.RandomState(3).randn(3, 5))).sum(), [(3, 4), (4, 5)]),
])
def test_op_gradients(name, fn, shapes):
    inputs = [rnd(*s) for s in shapes]
    assert gradcheck(fn, *inputs, verbose=True), f"gradient mismatch in {name}"


def test_grad_accumulates_when_tensor_used_twice():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x * x + x).sum()  # dy/dx = 2x + 1
    y.backward()
    assert np.allclose(x.grad, [5.0, 7.0])


def test_no_grad_builds_no_graph():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with nf.no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad
    assert y._prev == ()


def test_requires_grad_propagation():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0])
    assert (a + b).requires_grad
    assert not (b * b).requires_grad


def test_backward_on_non_scalar_requires_grad_argument():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
    (x * 2).backward(np.ones((1, 2)))
    assert np.allclose(x.grad, [[2.0, 2.0]])


def test_deep_graph_does_not_overflow_recursion():
    x = Tensor([1.0], requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert np.isfinite(x.grad).all()


def test_backward_frees_graph_without_cyclic_gc():
    """Regression: activations must be released as soon as the loss is dropped (no GC needed)."""
    import gc
    import weakref

    gc.disable()
    try:
        x = Tensor(np.random.randn(64, 64), requires_grad=True)
        h = (x @ x).tanh()
        loss = (h * h).sum()
        ref = weakref.ref(h)
        loss.backward()
        assert x.grad is not None
        assert loss._prev == () and ref() is not None  # still referenced by local `h`
        del h, loss
        assert ref() is None, "intermediate kept alive by a reference cycle"
    finally:
        gc.enable()


def test_graph_export():
    x = Tensor([1.0], requires_grad=True)
    y = (x * 2 + 1).sum()
    g = y.graph()
    assert len(g["nodes"]) >= 3
    assert any(n["op"] == "leaf" for n in g["nodes"])


def test_dtype_default_is_float32():
    assert Tensor([1.0, 2.0]).dtype == np.float32
    assert Tensor(np.array([1, 2])).dtype.kind == "i"
    with nf.default_dtype(np.float64):
        assert Tensor([1.0]).dtype == np.float64
