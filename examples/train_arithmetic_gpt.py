"""
Teach a GPT to add numbers - from scratch, in NumPy.

The model only ever sees strings like ``"482+017=0499;"``. It must learn
place value, carrying and the meaning of ``+`` and ``=`` purely from
next-token prediction. Accuracy is measured on *unseen* problems by prompting
with ``"482+017="`` and greedily decoding the answer.

    python examples/train_arithmetic_gpt.py --steps 2000 --dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neuroforge as nf  # noqa: E402
from neuroforge.data import CharTokenizer, make_arithmetic  # noqa: E402
from neuroforge.nn import GPT, GPTConfig  # noqa: E402
from neuroforge.optim import AdamW, CosineWarmup, clip_grad_norm_  # noqa: E402
from neuroforge.utils import Progress, count_parameters, save_checkpoint  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/arithmetic_gpt.npz")
    return p.parse_args()


def evaluate(model: GPT, tok: CharTokenizer, problems, n_digits: int, n: int = 200):
    """Greedy-decode the answer for ``n`` held-out problems; return accuracy and a few examples."""
    correct, examples = 0, []
    answer_len = n_digits + 1
    for line in problems[:n]:
        prompt, answer = line.split("=")
        prompt += "="
        ids = np.array([tok.encode(prompt)])
        out = model.generate(ids, max_new_tokens=answer_len, top_k=1)
        pred = tok.decode(out[0, len(prompt):])
        ok = pred == answer[:answer_len]
        correct += ok
        if len(examples) < 4:
            examples.append(f"{prompt}{pred}  {'ok' if ok else 'x  (want ' + answer[:answer_len] + ')'}")
    return correct / n, examples


def main():
    args = parse()
    nf.seed_everything(args.seed)

    lines = make_arithmetic(n=30000, n_digits=args.digits, seed=args.seed)
    tok = CharTokenizer("".join(lines))
    L = len(lines[0])
    data = np.array([tok.encode(line) for line in lines], dtype=np.int64)       # (N, L)
    n_train = int(0.95 * len(data))
    train, test_lines = data[:n_train], lines[n_train:]

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=L - 1, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=0.0)
    model = GPT(cfg)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineWarmup(opt, warmup_steps=100, total_steps=args.steps, min_lr=args.lr * 0.05)
    print(f"model: {count_parameters(model)} params | vocab {tok.vocab_size} | seq len {L} | train {len(train)} test {len(test_lines)}")

    dash = None
    if args.dashboard:
        from neuroforge.dashboard import Dashboard
        dash = Dashboard(port=args.port, title=f"{args.digits}-digit addition GPT", open_browser=True)
        dash.set_config(params=count_parameters(model), layers=cfg.n_layer, heads=cfg.n_head, d_model=cfg.n_embd,
                        batch=args.batch, steps=args.steps)
        dash.generate_fn = lambda prompt, n, temp: tok.decode(
            model.generate(np.array([tok.encode(prompt)]), max_new_tokens=min(n, L), temperature=temp, top_k=1)[0])

        def attention(prompt):
            ids = np.array([tok.encode(prompt[: cfg.block_size])])
            with nf.no_grad():
                model.eval(); model(ids); model.train()
            return {"tokens": [tok.itos[int(i)] for i in ids[0]],
                    "layers": [a[0].tolist() for a in model.attention_maps()]}
        dash.attention_fn = attention

    prog = Progress(args.steps, every=10)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ix = np.random.randint(0, len(train), size=args.batch)
        batch = train[ix]
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        gn = clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lr = sched.step()

        metrics = {"loss": float(loss.data), "lr": lr, "grad_norm": gn}
        if step % args.eval_every == 0 or step == args.steps:
            acc, examples = evaluate(model, tok, test_lines, args.digits)
            metrics["accuracy"] = acc
            print(f"\n[eval step {step}] accuracy {acc * 100:.1f}%  |  " + "  ".join(examples[:2]))
            if dash:
                dash.add_sample(step, "\n".join(examples), label=f"acc {acc * 100:.1f}%")
        if dash:
            dash.log(step, **metrics)
        prog.update(step, loss=float(loss.data), lr=lr)

    acc, examples = evaluate(model, tok, test_lines, args.digits, n=min(1000, len(test_lines)))
    print(f"\nfinal held-out accuracy: {acc * 100:.2f}%  ({time.time() - t0:.0f}s)")
    for e in examples:
        print("  ", e)
    save_checkpoint(args.out, model, opt, extra={"accuracy": acc, "chars": tok.chars})
    print(f"saved {args.out}")
    if dash:
        dash.finish()


if __name__ == "__main__":
    main()
