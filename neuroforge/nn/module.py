"""
neuroforge.nn.module
====================

``Module`` is the base class for every layer and model. It discovers
parameters by walking its attributes (tensors that require grad, child
modules, and lists/dicts of them), exactly like ``torch.nn.Module``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from ..tensor import Tensor


class Module:
    def __init__(self):
        self.training: bool = True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Parameter discovery
    # ------------------------------------------------------------------ #
    def _children(self) -> Iterator[Tuple[str, Any]]:
        for name, value in vars(self).items():
            if name == "training":
                continue
            yield name, value

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Tensor]]:
        seen = set()

        def walk(obj, pre):
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    yield pre, obj
            elif isinstance(obj, Module):
                for n, v in obj._children():
                    yield from walk(v, f"{pre}.{n}" if pre else n)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    yield from walk(v, f"{pre}.{i}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    yield from walk(v, f"{pre}.{k}")

        yield from walk(self, prefix)

    def parameters(self) -> List[Tensor]:
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for name, value in self._children():
            if isinstance(value, Module):
                yield from value.named_modules(f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, Module):
                        yield from v.named_modules(f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}")

    def modules(self) -> List["Module"]:
        return [m for _, m in self.named_modules()]

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    # ------------------------------------------------------------------ #
    # Training state
    # ------------------------------------------------------------------ #
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def state_dict(self) -> "OrderedDict[str, np.ndarray]":
        return OrderedDict((n, p.data.copy()) for n, p in self.named_parameters())

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"state_dict mismatch. missing={missing} unexpected={unexpected}")
        for k, v in state.items():
            if k in own:
                if own[k].shape != v.shape:
                    raise ValueError(f"shape mismatch for {k}: {own[k].shape} vs {v.shape}")
                own[k].data[...] = v

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = []
        for name, value in self._children():
            if isinstance(value, Module):
                lines.append(f"  ({name}): " + repr(value).replace("\n", "\n  "))
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, Module) for v in value):
                inner = "\n".join(f"    ({i}): " + repr(v).replace("\n", "\n    ") for i, v in enumerate(value))
                lines.append(f"  ({name}): [\n{inner}\n  ]")
        body = ("\n" + "\n".join(lines) + "\n") if lines else ""
        return f"{self.__class__.__name__}({self.extra_repr()}{body})"


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        self.items: List[Module] = list(modules)

    def append(self, m: Module) -> "ModuleList":
        self.items.append(m)
        return self

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)
