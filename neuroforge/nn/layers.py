"""
neuroforge.nn.layers
====================

Standard neural-network building blocks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import functional as F
from ..tensor import Tensor
from .module import Module


def _kaiming_uniform(fan_in: int, shape, gain: float = np.sqrt(2.0)) -> np.ndarray:
    bound = gain * np.sqrt(3.0 / fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


class Linear(Module):
    """``y = x @ W^T + b``. Weight shape is ``(out_features, in_features)``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Tensor(_kaiming_uniform(in_features, (out_features, in_features), gain=1.0), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        return out + self.bias if self.bias is not None else out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Tensor((np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32),
                             requires_grad=True)

    def forward(self, idx) -> Tensor:
        idx = idx.data if isinstance(idx, Tensor) else np.asarray(idx)
        return self.weight[idx.astype(np.int64)]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}"


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data.reshape(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.reshape(-1)
        else:
            mu, var = Tensor(self.running_mean), Tensor(self.running_var)
        return (x - mu) * ((var + self.eps) ** -0.5) * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = True):
        super().__init__()
        kh, kw = F._pair(kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = (kh, kw), stride, padding
        fan_in = in_channels * kh * kw
        self.weight = Tensor(_kaiming_uniform(fan_in, (out_channels, in_channels, kh, kw)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def extra_repr(self) -> str:
        return f"{self.kernel_size}"


class AvgPool2d(Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.kernel_size)


class MLP(Module):
    """Multi-layer perceptron: ``Linear -> act -> ... -> Linear``."""

    def __init__(self, sizes, activation: Optional[Module] = None, dropout: float = 0.0):
        super().__init__()
        act = activation or ReLU()
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(act)
                if dropout > 0:
                    self.layers.append(Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
