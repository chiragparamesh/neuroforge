"""
Train a character-level (or BPE) GPT on any text file.

    python examples/train_text_gpt.py --data data/shakespeare.txt --dashboard
    python examples/train_text_gpt.py --data data/shakespeare.txt --bpe 512 --n_layer 6 --n_embd 192

Press Ctrl+C at any time: the latest checkpoint is kept and you can sample from
it with ``examples/sample.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuroforge as nf  # noqa: E402
from neuroforge.data import BPETokenizer, CharTokenizer, TextDataset  # noqa: E402
from neuroforge.nn import GPT, GPTConfig  # noqa: E402
from neuroforge.optim import AdamW, CosineWarmup, clip_grad_norm_  # noqa: E402
from neuroforge.utils import Progress, count_parameters, save_checkpoint  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to a .txt file")
    p.add_argument("--bpe", type=int, default=0, help="train a byte-level BPE tokenizer with this vocab size (0 = char level)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=250)
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="checkpoints/text_gpt.npz")
    return p.parse_args()


def estimate_loss(model, ds, batch, iters=8):
    model.eval()
    losses = []
    with nf.no_grad():
        for _ in range(iters):
            x, y = ds.get_batch(batch, "val")
            _, loss = model(x, y)
            losses.append(float(loss.data))
    model.train()
    return float(np.mean(losses))


def main():
    args = parse()
    nf.seed_everything(args.seed)
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} characters")

    if args.bpe:
        tok = BPETokenizer().train(text[:500_000], vocab_size=args.bpe, verbose=True)
        tok_path = os.path.splitext(args.out)[0] + ".tokenizer.json"
    else:
        tok = CharTokenizer(text)
        tok_path = os.path.splitext(args.out)[0] + ".tokenizer.json"
    ids = tok.encode(text)
    ds = TextDataset(ids, args.block)
    print(f"vocab {tok.vocab_size} | tokens {len(ids):,} | train {len(ds.train_ids):,} val {len(ds.val_ids):,}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = CosineWarmup(opt, warmup_steps=min(200, args.steps // 10), total_steps=args.steps, min_lr=args.lr * 0.1)
    print(f"model: {count_parameters(model)} params\n{model}")

    def sample(prompt="\n", n=200, temp=0.8, top_k=40):
        ids_ = np.array([tok.encode(prompt) or [0]])
        return tok.decode(model.generate(ids_, max_new_tokens=n, temperature=temp, top_k=top_k)[0])

    dash = None
    if args.dashboard:
        from neuroforge.dashboard import Dashboard
        dash = Dashboard(port=args.port, title=os.path.basename(args.data), open_browser=True)
        dash.set_config(params=count_parameters(model), layers=cfg.n_layer, heads=cfg.n_head, d_model=cfg.n_embd,
                        context=cfg.block_size, vocab=tok.vocab_size, tokenizer="BPE" if args.bpe else "char")
        dash.generate_fn = lambda prompt, n, temp: sample(prompt or "\n", n, temp)

        def attention(prompt):
            ids_ = np.array([tok.encode(prompt)[: cfg.block_size] or [0]])
            with nf.no_grad():
                model.eval(); model(ids_); model.train()
            return {"tokens": [tok.decode([int(i)]) for i in ids_[0]],
                    "layers": [a[0].tolist() for a in model.attention_maps()]}
        dash.attention_fn = attention

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tok.save(tok_path)
    prog = Progress(args.steps, every=10)
    best_val, t0 = float("inf"), time.time()
    try:
        for step in range(1, args.steps + 1):
            x, y = ds.get_batch(args.batch, "train")
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            gn = clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr = sched.step()
            metrics = {"loss": float(loss.data), "lr": lr, "grad_norm": gn}
            if step % args.eval_every == 0 or step == args.steps:
                val = estimate_loss(model, ds, args.batch)
                metrics["val_loss"] = val
                if val < best_val:
                    best_val = val
                    save_checkpoint(args.out, model, opt, extra={"step": step, "val_loss": val, "tokenizer": tok_path})
            if step % args.sample_every == 0:
                s = sample()
                print(f"\n--- sample @ step {step} ---\n{s}\n")
                if dash:
                    dash.add_sample(step, s)
            if dash:
                dash.log(step, **metrics)
            prog.update(step, loss=float(loss.data), lr=lr)
    except KeyboardInterrupt:
        print("\ninterrupted - keeping best checkpoint")
    print(f"\nbest val loss {best_val:.4f} | {time.time() - t0:.0f}s | checkpoint {args.out}")
    print(sample(n=400))
    if dash:
        dash.finish()


if __name__ == "__main__":
    main()
