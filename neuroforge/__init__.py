"""
NeuroForge
==========

A deep learning framework built from scratch in pure NumPy.

    >>> import neuroforge as nf
    >>> x = nf.randn(4, 3, requires_grad=True)
    >>> y = (x ** 2).sum()
    >>> y.backward()
    >>> x.grad            # == 2 * x

Sub-packages
------------
* ``neuroforge.tensor``      - Tensor + reverse-mode autograd
* ``neuroforge.functional``  - activations, losses, conv, attention
* ``neuroforge.nn``          - layers, Module, GPT transformer
* ``neuroforge.optim``       - SGD, Adam, AdamW, schedulers
* ``neuroforge.data``        - tokenisers (char, BPE), datasets
* ``neuroforge.dashboard``   - live browser training dashboard
"""

from .tensor import (
    Tensor, no_grad, default_dtype, is_grad_enabled, tensor, zeros, ones, randn, rand, arange, eye,
    cat, stack, where, tril,
)
from . import functional
from . import nn
from . import optim
from . import data
from . import utils
from .utils import seed_everything, gradcheck, save_checkpoint, load_checkpoint

__version__ = "0.1.0"
__all__ = [
    "Tensor", "no_grad", "default_dtype", "is_grad_enabled", "tensor", "zeros", "ones", "randn", "rand", "arange", "eye",
    "cat", "stack", "where", "tril", "functional", "nn", "optim", "data", "utils",
    "seed_everything", "gradcheck", "save_checkpoint", "load_checkpoint", "__version__",
]
