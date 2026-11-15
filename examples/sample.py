"""
Sample text from a checkpoint produced by ``train_text_gpt.py``.

    python examples/sample.py --ckpt checkpoints/text_gpt.npz --prompt "ROMEO:" --tokens 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuroforge.data import BPETokenizer, CharTokenizer  # noqa: E402
from neuroforge.nn import GPT, GPTConfig  # noqa: E402
from neuroforge.utils import load_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/text_gpt.npz")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    meta = load_checkpoint(args.ckpt)
    cfg = GPTConfig(**meta["config"])
    model = GPT(cfg)
    model.load_state_dict(meta["state_dict"])
    tok_path = meta["extra"]["tokenizer"]
    with open(tok_path, encoding="utf-8") as f:
        kind = json.load(f)["type"]
    tok = BPETokenizer.load(tok_path) if kind == "bpe" else CharTokenizer.load(tok_path)

    ids = np.array([tok.encode(args.prompt) or [0]])
    out = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
