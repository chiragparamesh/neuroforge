"""
neuroforge.functional
=====================

Stateless differentiable functions: activations, losses, convolution, pooling
and attention helpers. Every function here either composes existing Tensor
ops (and therefore gets autograd for free) or defines its own backward pass.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .tensor import Tensor, _GradMode, _as_tensor, tril, where

IntPair = Union[int, Tuple[int, int]]


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


def dropout(x: Tensor, p: float, training: bool = True) -> Tensor:
    """Inverted dropout: scales kept activations by 1/(1-p) at train time."""
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.shape) >= p).astype(np.float32) / (1.0 - p)
    return x * Tensor(mask)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def cross_entropy(logits: Tensor, targets, ignore_index: Optional[int] = None,
                  label_smoothing: float = 0.0) -> Tensor:
    """Mean cross-entropy over all leading dims.

    ``logits``: ``(..., C)``; ``targets``: integer array of shape ``(...)``.
    """
    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    t = t.astype(np.int64).reshape(-1)
    C = logits.shape[-1]
    flat = logits.reshape(-1, C)
    logp = flat.log_softmax(axis=-1)

    if ignore_index is not None:
        valid = t != ignore_index
        t_safe = np.where(valid, t, 0)
        n = max(int(valid.sum()), 1)
        weights = valid.astype(np.float32) / n
    else:
        t_safe = t
        n = t.shape[0]
        weights = np.full(t.shape, 1.0 / n, dtype=np.float32)

    picked = logp[np.arange(t.shape[0]), t_safe]  # (N,)
    nll = -(picked * Tensor(weights)).sum()
    if label_smoothing > 0.0:
        smooth = -(logp.mean(axis=-1) * Tensor(weights)).sum()
        return nll * (1.0 - label_smoothing) + smooth * label_smoothing
    return nll


def nll_loss(logp: Tensor, targets) -> Tensor:
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).astype(np.int64).reshape(-1)
    flat = logp.reshape(-1, logp.shape[-1])
    return -flat[np.arange(t.shape[0]), t].mean()


def mse_loss(pred: Tensor, target) -> Tensor:
    target = _as_tensor(target)
    return ((pred - target) ** 2).mean()


def l1_loss(pred: Tensor, target) -> Tensor:
    return (pred - _as_tensor(target)).abs().mean()


def binary_cross_entropy_with_logits(logits: Tensor, target) -> Tensor:
    """Numerically stable BCE: max(x,0) - x*y + log(1 + exp(-|x|))."""
    y = _as_tensor(target)
    x = logits
    return (x.clip(0, np.inf) - x * y + (1.0 + (-x.abs()).exp()).log()).mean()


def huber_loss(pred: Tensor, target, delta: float = 1.0) -> Tensor:
    diff = pred - _as_tensor(target)
    absd = diff.abs()
    quad = 0.5 * diff ** 2
    lin = delta * (absd - 0.5 * delta)
    return where(absd.data <= delta, quad, lin).mean()


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def layer_norm(x: Tensor, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (x - mu) * ((var + eps) ** -0.5)
    if weight is not None:
        xhat = xhat * weight
    if bias is not None:
        xhat = xhat + bias
    return xhat


# --------------------------------------------------------------------------- #
# Convolution / pooling (im2col based)
# --------------------------------------------------------------------------- #
def _pair(v: IntPair) -> Tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _im2col(xp: np.ndarray, KH: int, KW: int, sh: int, sw: int, OH: int, OW: int) -> np.ndarray:
    """``xp``: padded input ``(N, C, Hp, Wp)`` -> ``(N, C*KH*KW, OH*OW)``."""
    N, C, _, _ = xp.shape
    sN, sC, sH, sW = xp.strides
    windows = as_strided(
        xp,
        shape=(N, C, KH, KW, OH, OW),
        strides=(sN, sC, sH, sW, sH * sh, sW * sw),
        writeable=False,
    )
    return windows.reshape(N, C * KH * KW, OH * OW)


def _col2im(dcols: np.ndarray, xp_shape, KH: int, KW: int, sh: int, sw: int, OH: int, OW: int) -> np.ndarray:
    N, C, Hp, Wp = xp_shape
    dxp = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    d6 = dcols.reshape(N, C, KH, KW, OH, OW)
    for kh in range(KH):
        for kw in range(KW):
            dxp[:, :, kh: kh + sh * OH: sh, kw: kw + sw * OW: sw] += d6[:, :, kh, kw]
    return dxp


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: IntPair = 1, padding: IntPair = 0) -> Tensor:
    """2-D convolution (cross-correlation) over ``(N, C, H, W)`` input."""
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    N, C, H, W = x.shape
    F, Cw, KH, KW = weight.shape
    assert C == Cw, f"input channels {C} != weight channels {Cw}"
    OH = (H + 2 * ph - KH) // sh + 1
    OW = (W + 2 * pw - KW) // sw + 1

    xp = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = _im2col(xp, KH, KW, sh, sw, OH, OW)                  # (N, CKK, L)
    w2 = weight.data.reshape(F, -1)                             # (F, CKK)
    out_data = np.matmul(w2, cols).reshape(N, F, OH, OW)        # (N, F, OH, OW)
    if bias is not None:
        out_data = out_data + bias.data.reshape(1, F, 1, 1)

    children = (x, weight) + ((bias,) if bias is not None else ())
    req = _GradMode.enabled and any(c.requires_grad for c in children)
    out = Tensor(out_data, requires_grad=req, _children=children if req else (), _op="conv2d" if req else "")

    def _backward():
        g2 = out.grad.reshape(N, F, OH * OW)                    # (N, F, L)
        if weight.requires_grad:
            dw = np.einsum("nfl,nkl->fk", g2, cols).reshape(weight.shape)
            weight._accumulate(dw)
        if bias is not None and bias.requires_grad:
            bias._accumulate(out.grad.sum(axis=(0, 2, 3)))
        if x.requires_grad:
            dcols = np.matmul(w2.T, g2)                         # (N, CKK, L)
            dxp = _col2im(dcols, xp.shape, KH, KW, sh, sw, OH, OW)
            x._accumulate(dxp[:, :, ph: ph + H, pw: pw + W])

    out._backward = _backward
    return out


def max_pool2d(x: Tensor, kernel_size: IntPair, stride: Optional[IntPair] = None) -> Tensor:
    """Max pooling. ``stride`` defaults to ``kernel_size`` (non-overlapping windows)."""
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride) if stride is not None else (kh, kw)
    N, C, H, W = x.shape
    OH = (H - kh) // sh + 1
    OW = (W - kw) // sw + 1
    sN, sC, sH, sW = x.data.strides
    windows = as_strided(x.data, shape=(N, C, OH, OW, kh, kw),
                         strides=(sN, sC, sH * sh, sW * sw, sH, sW), writeable=False)
    flat = windows.reshape(N, C, OH, OW, kh * kw)
    argmax = flat.argmax(axis=-1)
    out_data = np.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]
    req = _GradMode.enabled and x.requires_grad
    out = Tensor(out_data, requires_grad=req, _children=(x,) if req else (), _op="max_pool2d" if req else "")

    def _backward():
        if x.requires_grad:
            dx = np.zeros_like(x.data, dtype=np.float32)
            n, c, oh, ow = np.indices((N, C, OH, OW))
            hi = oh * sh + argmax // kw
            wi = ow * sw + argmax % kw
            np.add.at(dx, (n, c, hi, wi), out.grad)
            x._accumulate(dx)

    out._backward = _backward
    return out


def avg_pool2d(x: Tensor, kernel_size: IntPair) -> Tensor:
    kh, kw = _pair(kernel_size)
    N, C, H, W = x.shape
    OH, OW = H // kh, W // kw
    x = x[:, :, : OH * kh, : OW * kw]
    return x.reshape(N, C, OH, kh, OW, kw).mean(axis=(3, 5))


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
                                 dropout_p: float = 0.0, training: bool = False,
                                 return_weights: bool = False):
    """``q, k, v``: ``(B, nh, T, hd)``. Returns ``(B, nh, T, hd)`` and optionally the weights."""
    hd = q.shape[-1]
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(hd))     # (B, nh, T, T)
    if causal:
        T = q.shape[-2]
        mask = ~tril(T)
        att = att.masked_fill(mask, -1e9)
    att = att.softmax(axis=-1)
    att = dropout(att, dropout_p, training)
    y = att @ v
    return (y, att) if return_weights else y


def one_hot(indices, num_classes: int) -> Tensor:
    idx = np.asarray(indices.data if isinstance(indices, Tensor) else indices).astype(np.int64)
    return Tensor(np.eye(num_classes, dtype=np.float32)[idx])
