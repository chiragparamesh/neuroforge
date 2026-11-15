"""
A tour of the autograd engine: build an expression, inspect the graph, run
backward, and compare with hand-derived gradients.

    python examples/autograd_demo.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuroforge as nf  # noqa: E402


def main():
    # f(x, w) = sum( tanh(x @ w) ** 2 )
    x = nf.Tensor([[1.0, -2.0], [0.5, 0.25]], requires_grad=True)
    w = nf.Tensor([[0.3, -0.1], [0.8, 0.5]], requires_grad=True)
    x.name, w.name = "x", "w"

    h = (x @ w).tanh()
    f = (h ** 2).sum()
    f.backward()

    print("f =", f.item())
    print("df/dx =\n", x.grad)
    print("df/dw =\n", w.grad)

    # Hand derivation: df/dh = 2h ; dh/dz = 1 - h^2 ; dz/dx = w^T ; dz/dw = x^T
    z = x.data @ w.data
    hh = np.tanh(z)
    dz = 2 * hh * (1 - hh ** 2)
    print("hand df/dx matches:", np.allclose(x.grad, dz @ w.data.T, atol=1e-6))
    print("hand df/dw matches:", np.allclose(w.grad, x.data.T @ dz, atol=1e-6))

    print("\nComputation graph:")
    g = f.graph()
    for n in g["nodes"]:
        print(f"  node {n['id']:2d}  op={n['op']:<10} shape={n['shape']}  name={n['name']}")
    print("  edges:", json.dumps(g["edges"]))

    # Numerical gradient check on a trickier function
    ok = nf.gradcheck(lambda a, b: ((a @ b).softmax(-1).log() * nf.Tensor([[1.0, 2.0], [3.0, 4.0]])).sum(), x, w)
    print("\ngradcheck (softmax -> log -> weighted sum):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
