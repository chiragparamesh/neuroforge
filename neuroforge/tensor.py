"""
neuroforge.tensor
=================

The core of NeuroForge: a NumPy-backed ``Tensor`` with reverse-mode automatic
differentiation.

Every differentiable operation creates a new ``Tensor`` that remembers its
parents and a small closure (``_backward``) describing how to push gradients
to them. Calling ``.backward()`` on a scalar walks the graph in reverse
topological order and accumulates ``.grad`` on every tensor that asked for it.

Design notes
------------
* Data is always stored as ``float32`` unless the tensor is used purely as an
  index/mask holder, in which case integer/bool dtypes are preserved.
* Broadcasting is fully supported. ``_unbroadcast`` sums a gradient back down
  to the shape of the operand it belongs to.
* ``no_grad()`` disables graph construction (used for inference/generation).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]
TensorLike = Union["Tensor", np.ndarray, Number, list]


# --------------------------------------------------------------------------- #
# Global autograd state
# --------------------------------------------------------------------------- #
class _GradMode:
    enabled: bool = True
    default_dtype = np.float32


@contextlib.contextmanager
def no_grad():
    """Context manager that disables gradient tracking.

    >>> with no_grad():
    ...     y = model(x)   # no graph is built, memory is saved
    """
    prev = _GradMode.enabled
    _GradMode.enabled = False
    try:
        yield
    finally:
        _GradMode.enabled = prev


@contextlib.contextmanager
def default_dtype(dtype):
    """Temporarily change the floating dtype new tensors are created with (e.g. float64 for gradcheck)."""
    prev = _GradMode.default_dtype
    _GradMode.default_dtype = np.dtype(dtype).type
    try:
        yield
    finally:
        _GradMode.default_dtype = prev


def is_grad_enabled() -> bool:
    return _GradMode.enabled


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` so that it has ``shape`` (undoing NumPy broadcasting)."""
    if grad.shape == shape:
        return grad
    # Sum over leading dimensions that were added by broadcasting.
    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    # Sum over dimensions that were 1 in the original shape.
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _as_tensor(x: TensorLike) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


def _noop() -> None:
    return None


def _blas_ready(a: np.ndarray) -> np.ndarray:
    """Return ``a`` in a layout NumPy can hand to BLAS.

    ``np.matmul`` only calls BLAS when each inner 2-D matrix is C- or
    F-contiguous; otherwise it silently falls back to a loop that is ~50x
    slower. Transposed views (e.g. attention heads) trip this, so we copy them
    to contiguous memory first. The copy is far cheaper than the slow path.
    """
    if a.ndim < 2:
        return a
    it = a.itemsize
    rows, cols = a.shape[-2:]
    s_r, s_c = a.strides[-2:]
    inner_c = s_c == it and s_r == it * cols
    inner_f = s_r == it and s_c == it * rows
    if inner_c or inner_f or rows == 1 or cols == 1:
        return a
    return np.ascontiguousarray(a)


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``np.matmul`` with a fast path for the ``(..., K) @ (K, N)`` case.

    NumPy's batched matmul runs one small GEMM per batch element, which is
    ~20x slower than a single big GEMM on the flattened input. Linear layers
    hit this case on every call, so it matters a lot.
    """
    if a.ndim > 2 and b.ndim == 2:
        K = a.shape[-1]
        return np.matmul(a.reshape(-1, K), b).reshape(a.shape[:-1] + (b.shape[-1],))
    return np.matmul(a, b)


# --------------------------------------------------------------------------- #
# Tensor
# --------------------------------------------------------------------------- #
class Tensor:
    """A multi-dimensional array that supports automatic differentiation."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op", "name", "__weakref__")
    __array_priority__ = 1000  # make NumPy defer to our __radd__ etc.

    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
        dtype=None,
    ):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype)
        elif arr.dtype.kind in "fc":
            if arr.dtype != _GradMode.default_dtype:
                arr = arr.astype(_GradMode.default_dtype)
        elif arr.dtype.kind in "iub":
            pass  # keep integer / bool arrays as-is (indices, masks)
        else:
            arr = arr.astype(_GradMode.default_dtype)
        self.data: np.ndarray = arr
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = bool(requires_grad)
        self._backward: Callable[[], None] = _noop
        self._prev: Tuple["Tensor", ...] = tuple(_children) if _GradMode.enabled else ()
        self._op: str = _op if _GradMode.enabled else ""
        self.name: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, suppress_small=True)}{grad_str})"

    def item(self) -> float:
        return float(self.data.reshape(-1)[0]) if self.data.size == 1 else float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self):
        return self.data.tolist()

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        self.requires_grad = flag
        return self

    # ------------------------------------------------------------------ #
    # Graph construction helper
    # ------------------------------------------------------------------ #
    def _make(self, data: np.ndarray, children: Sequence["Tensor"], op: str) -> "Tensor":
        """Create an output tensor whose requires_grad is inherited from children."""
        req = _GradMode.enabled and any(c.requires_grad for c in children)
        out = Tensor(data, requires_grad=req, _children=children if req else (), _op=op if req else "")
        return out

    # ------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------ #
    def __add__(self, other: TensorLike) -> "Tensor":
        other = _as_tensor(other)
        out = self._make(self.data + other.data, (self, other), "add")

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: TensorLike) -> "Tensor":
        other = _as_tensor(other)
        out = self._make(self.data * other.data, (self, other), "mul")

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, power: Number) -> "Tensor":
        assert isinstance(power, (int, float)), "only scalar powers are supported"
        out = self._make(self.data ** power, (self,), f"pow{power}")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * power * self.data ** (power - 1))

        out._backward = _backward
        return out

    def __matmul__(self, other: TensorLike) -> "Tensor":
        other = _as_tensor(other)
        a, b = _blas_ready(self.data), _blas_ready(other.data)
        out = self._make(_matmul(a, b), (self, other), "matmul")

        def _backward():
            g = _blas_ready(out.grad)
            if self.requires_grad:
                if b.ndim == 1:
                    ga = np.multiply.outer(g, b) if g.ndim >= 1 else g * b
                else:
                    ga = _matmul(g, _blas_ready(np.swapaxes(b, -1, -2)))
                self._accumulate(_unbroadcast(ga, self.shape))
            if other.requires_grad:
                if a.ndim == 1:
                    gb = np.multiply.outer(a, g) if g.ndim >= 1 else a * g
                elif a.ndim > 2 and b.ndim == 2:
                    # (..., K) @ (K, N): one big GEMM gives dW directly, no batch reduction needed.
                    K, N = a.shape[-1], g.shape[-1]
                    gb = np.matmul(a.reshape(-1, K).T, g.reshape(-1, N))
                else:
                    gb = np.matmul(_blas_ready(np.swapaxes(a, -1, -2)), g)
                other._accumulate(_unbroadcast(gb, other.shape))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + (-_as_tensor(other))

    def __truediv__(self, other: TensorLike) -> "Tensor":
        return self * (_as_tensor(other) ** -1.0)

    def __radd__(self, other: TensorLike) -> "Tensor":
        return _as_tensor(other) + self

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return _as_tensor(other) - self

    def __rmul__(self, other: TensorLike) -> "Tensor":
        return _as_tensor(other) * self

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        return _as_tensor(other) / self

    def __rmatmul__(self, other: TensorLike) -> "Tensor":
        return _as_tensor(other) @ self

    # Comparisons return plain (non-differentiable) tensors.
    def __gt__(self, other):
        return Tensor(self.data > _as_tensor(other).data)

    def __lt__(self, other):
        return Tensor(self.data < _as_tensor(other).data)

    def __ge__(self, other):
        return Tensor(self.data >= _as_tensor(other).data)

    def __le__(self, other):
        return Tensor(self.data <= _as_tensor(other).data)

    def __eq__(self, other):  # type: ignore[override]
        return Tensor(self.data == _as_tensor(other).data)

    __hash__ = object.__hash__

    # ------------------------------------------------------------------ #
    # Element-wise functions
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        out = self._make(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = self._make(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def abs(self) -> "Tensor":
        out = self._make(np.abs(self.data), (self,), "abs")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * np.sign(self.data))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def leaky_relu(self, slope: float = 0.01) -> "Tensor":
        out = self._make(np.where(self.data > 0, self.data, self.data * slope), (self,), "leaky_relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * np.where(self.data > 0, 1.0, slope).astype(np.float32))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._make(s, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = self._make(t, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """GELU activation (tanh approximation, as used in GPT-2)."""
        x = self.data
        c = x.dtype.type(np.sqrt(2.0 / np.pi))
        k = x.dtype.type(0.044715)
        x2 = x * x
        t = np.tanh(c * (x + k * x2 * x))
        out = self._make(0.5 * x * (1.0 + t), (self,), "gelu")

        def _backward():
            if self.requires_grad:
                dinner = c * (1.0 + 3 * k * x2)
                grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
                self._accumulate(out.grad * grad)

        out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        return self * self.sigmoid()

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = self._make(s, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                g = out.grad
                self._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        ls = shifted - lse
        out = self._make(ls, (self,), "log_softmax")

        def _backward():
            if self.requires_grad:
                g = out.grad
                self._accumulate(g - np.exp(ls) * g.sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def clip(self, lo: float, hi: float) -> "Tensor":
        out = self._make(np.clip(self.data, lo, hi), (self,), "clip")

        def _backward():
            if self.requires_grad:
                mask = (self.data >= lo) & (self.data <= hi)
                self._accumulate(out.grad * mask)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._accumulate(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims: bool = False, unbiased: bool = False) -> "Tensor":
        mu = self.mean(axis=axis, keepdims=True)
        sq = (self - mu) ** 2
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = int(np.prod([self.shape[a] for a in axes]))
        denom = (n - 1) if unbiased else n
        return sq.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def std(self, axis=None, keepdims: bool = False, eps: float = 0.0) -> "Tensor":
        return (self.var(axis=axis, keepdims=keepdims) + eps) ** 0.5

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        m = self.data.max(axis=axis, keepdims=keepdims)
        out = self._make(m, (self,), "max")

        def _backward():
            if self.requires_grad:
                m_k = self.data.max(axis=axis, keepdims=True)
                mask = (self.data == m_k).astype(np.float32)
                mask /= mask.sum(axis=axis, keepdims=True)  # split ties evenly
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._accumulate(mask * g)

        out._backward = _backward
        return out

    def min(self, axis=None, keepdims: bool = False) -> "Tensor":
        return -((-self).max(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------------ #
    # Shape manipulation
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    view = reshape

    def flatten(self, start_dim: int = 1) -> "Tensor":
        new_shape = self.shape[:start_dim] + (-1,)
        return self.reshape(new_shape)

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 2:
            # torch-style transpose(dim0, dim1): swap two axes
            perm = list(range(self.ndim))
            perm[axes[0]], perm[axes[1]] = perm[axes[1]], perm[axes[0]]
            axes = tuple(perm)
        out = self._make(self.data.transpose(axes), (self,), "transpose")
        inv = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.transpose(inv))

        out._backward = _backward
        return out

    permute = transpose

    def swapaxes(self, a: int, b: int) -> "Tensor":
        perm = list(range(self.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.transpose(tuple(perm))

    def expand(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(np.broadcast_to(self.data, shape).copy(), (self,), "expand")

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))

        out._backward = _backward
        return out

    def unsqueeze(self, axis: int) -> "Tensor":
        if axis < 0:
            axis = self.ndim + axis + 1
        return self.reshape(self.shape[:axis] + (1,) + self.shape[axis:])

    def squeeze(self, axis: Optional[int] = None) -> "Tensor":
        return self.reshape(np.squeeze(self.data, axis=axis).shape)

    def __getitem__(self, idx) -> "Tensor":
        if isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        out = self._make(self.data[idx], (self,), "index")

        def _backward():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                np.add.at(g, idx, out.grad)
                self._accumulate(g)

        out._backward = _backward
        return out

    def masked_fill(self, mask: TensorLike, value: float) -> "Tensor":
        """Return a copy where positions with ``mask == True`` are set to ``value``."""
        m = _as_tensor(mask).data.astype(bool)
        out = self._make(np.where(m, np.float32(value), self.data), (self,), "masked_fill")

        def _backward():
            if self.requires_grad:
                self._accumulate(np.where(m, 0.0, out.grad).astype(np.float32))

        out._backward = _backward
        return out

    def pad(self, pad_width, value: float = 0.0) -> "Tensor":
        """Pad with constant ``value``; ``pad_width`` follows ``np.pad`` conventions."""
        out = self._make(np.pad(self.data, pad_width, constant_values=value), (self,), "pad")

        def _backward():
            if self.requires_grad:
                slices = tuple(slice(lo, out.shape[i] - hi) for i, (lo, hi) in enumerate(pad_width))
                self._accumulate(out.grad[slices])

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Autograd
    # ------------------------------------------------------------------ #
    def _accumulate(self, g: np.ndarray) -> None:
        g = np.asarray(g, dtype=self.data.dtype if self.data.dtype.kind == "f" else np.float32)
        if g.shape != self.shape:
            g = _unbroadcast(g, self.shape)
        if self.grad is None:
            self.grad = g.copy() if g.base is not None else g
        else:
            self.grad = self.grad + g

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Run reverse-mode autodiff from this tensor.

        If the tensor is not a scalar you must pass ``grad`` (dL/dself).
        """
        if not self.requires_grad:
            raise RuntimeError("called backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=self.data.dtype).reshape(self.shape)

        # Build topological order (iterative DFS so deep graphs don't overflow the stack).
        topo: list = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        for node in reversed(topo):
            node._backward()

        # Tear the graph down immediately. Each node's _backward closure refers back
        # to the node itself (a reference cycle), so without this the whole graph -
        # every activation of the forward pass - would linger until Python's cyclic
        # GC happens to run. NumPy buffers do not count toward GC thresholds, so that
        # can take many training steps and exhaust memory. Like PyTorch, we also drop
        # the gradients of non-leaf intermediates; only leaves keep .grad.
        for node in topo:
            if node._prev:
                node._backward = _noop
                node._prev = ()
                if node is not self:
                    node.grad = None

    # ------------------------------------------------------------------ #
    # Graph introspection (used by the dashboard / debugging)
    # ------------------------------------------------------------------ #
    def graph(self) -> dict:
        """Return a JSON-serialisable description of the computation graph."""
        nodes, edges, seen = [], [], {}

        def visit(t: "Tensor"):
            if id(t) in seen:
                return seen[id(t)]
            nid = len(nodes)
            seen[id(t)] = nid
            nodes.append({"id": nid, "op": t._op or ("leaf" if t.requires_grad else "const"),
                          "shape": list(t.shape), "name": t.name})
            for c in t._prev:
                edges.append({"from": visit(c), "to": nid})
            return nid

        visit(self)
        return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
# Factory functions
# --------------------------------------------------------------------------- #
def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False, std: float = 1.0) -> Tensor:
    return Tensor((np.random.randn(*shape) * std).astype(np.float32), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def arange(*args, requires_grad: bool = False) -> Tensor:
    return Tensor(np.arange(*args, dtype=np.float32), requires_grad=requires_grad)


def eye(n: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.eye(n, dtype=np.float32), requires_grad=requires_grad)


def cat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along ``axis``."""
    tensors = [_as_tensor(t) for t in tensors]
    data = np.concatenate([t.data for t in tensors], axis=axis)
    req = _GradMode.enabled and any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=req, _children=tensors if req else (), _op="cat" if req else "")
    sizes = [t.shape[axis] for t in tensors]

    def _backward():
        offset = 0
        for t, s in zip(tensors, sizes):
            if t.requires_grad:
                sl = [slice(None)] * out.ndim
                sl[axis] = slice(offset, offset + s)
                t._accumulate(out.grad[tuple(sl)])
            offset += s

    out._backward = _backward
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return cat([t.unsqueeze(axis) for t in tensors], axis=axis)


def where(cond: TensorLike, a: TensorLike, b: TensorLike) -> Tensor:
    cond_np = _as_tensor(cond).data.astype(bool)
    a, b = _as_tensor(a), _as_tensor(b)
    data = np.where(cond_np, a.data, b.data)
    req = _GradMode.enabled and (a.requires_grad or b.requires_grad)
    out = Tensor(data, requires_grad=req, _children=(a, b) if req else (), _op="where" if req else "")

    def _backward():
        if a.requires_grad:
            a._accumulate(_unbroadcast(np.where(cond_np, out.grad, 0.0), a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(np.where(cond_np, 0.0, out.grad), b.shape))

    out._backward = _backward
    return out


def tril(n: int) -> np.ndarray:
    """Lower-triangular ``(n, n)`` boolean mask (used for causal attention)."""
    return np.tril(np.ones((n, n), dtype=bool))
