"""
neuroforge.cli
==============

``neuroforge info``   - environment and self-test summary
``neuroforge bench``  - micro-benchmarks (matmul, conv, GPT forward/backward)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from . import __version__


def _info(_args) -> int:
    import neuroforge as nf
    from neuroforge.nn import GPT, GPTConfig

    print(f"NeuroForge {__version__}")
    print(f"Python     {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"NumPy      {np.__version__}")
    x = nf.randn(8, 8, requires_grad=True)
    ((x ** 2).sum()).backward()
    ok = np.allclose(x.grad, 2 * x.data)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    idx = np.random.randint(0, 32, (2, 16))
    _, loss = m(idx, idx)
    loss.backward()
    grads = all(p.grad is not None for p in m.parameters())
    print(f"Self-test  autograd={'ok' if ok else 'FAIL'}  gpt_backward={'ok' if grads else 'FAIL'}")
    return 0 if ok and grads else 1


def _bench(args) -> int:
    import neuroforge as nf
    import neuroforge.functional as F
    from neuroforge.nn import GPT, GPTConfig

    def timeit(fn, n=5):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    print(f"{'benchmark':<40}{'ms/iter':>10}")
    print("-" * 50)
    a, b = nf.randn(512, 512, requires_grad=True), nf.randn(512, 512, requires_grad=True)
    print(f"{'matmul 512x512 fwd+bwd':<40}{timeit(lambda: (a @ b).sum().backward()) * 1e3:>10.2f}")
    img, k = nf.randn(16, 3, 32, 32, requires_grad=True), nf.randn(16, 3, 3, 3, requires_grad=True)
    print(f"{'conv2d 16x3x32x32 -> 16ch fwd+bwd':<40}{timeit(lambda: F.conv2d(img, k, padding=1).sum().backward()) * 1e3:>10.2f}")
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    m = GPT(cfg)
    idx = np.random.randint(0, 256, (args.batch, 64))

    def step():
        _, loss = m(idx, idx)
        loss.backward()
        m.zero_grad()

    ms = timeit(step, n=3) * 1e3
    toks = args.batch * 64
    print(f"{'GPT 4L/128d B=' + str(args.batch) + ' T=64 fwd+bwd':<40}{ms:>10.2f}   ({toks / ms * 1e3:,.0f} tok/s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neuroforge", description="NeuroForge: deep learning from scratch in NumPy")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print environment info and run a self-test")
    b = sub.add_parser("bench", help="run micro-benchmarks")
    b.add_argument("--batch", type=int, default=16)
    args = p.parse_args(argv)
    if args.cmd == "info":
        return _info(args)
    if args.cmd == "bench":
        return _bench(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
