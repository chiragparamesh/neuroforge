"""The transformer: shapes, causality, overfitting a batch, generation, checkpoints."""

import os

import numpy as np

import neuroforge as nf
from neuroforge.nn import GPT, GPTConfig
from neuroforge.optim import AdamW
from neuroforge.utils import load_checkpoint, save_checkpoint


def small_cfg(**kw):
    base = dict(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes_and_loss():
    m = GPT(small_cfg())
    idx = np.random.randint(0, 17, (3, 8))
    logits = m(idx)
    assert logits.shape == (3, 8, 17)
    logits, loss = m(idx, idx)
    assert loss.shape == ()
    assert np.isclose(loss.item(), np.log(17), atol=0.5)  # near-uniform at init


def test_tied_weights_share_parameter():
    m = GPT(small_cfg(tie_weights=True))
    assert m.lm_head.weight is m.tok_emb.weight
    names = [n for n, _ in m.named_parameters()]
    assert names.count("tok_emb.weight") == 1 and "lm_head.weight" not in names
    m2 = GPT(small_cfg(tie_weights=False))
    assert m2.num_parameters() == m.num_parameters() + 17 * 16


def test_causality_future_tokens_do_not_affect_past_logits():
    m = GPT(small_cfg())
    a = np.random.randint(0, 17, (1, 8))
    b = a.copy()
    b[0, 5:] = (b[0, 5:] + 1) % 17  # change only the future
    la, lb = m(a).data, m(b).data
    assert np.allclose(la[0, :5], lb[0, :5], atol=1e-5)
    assert not np.allclose(la[0, 5:], lb[0, 5:])


def test_all_parameters_receive_gradients():
    m = GPT(small_cfg())
    idx = np.random.randint(0, 17, (2, 8))
    _, loss = m(idx, idx)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and np.isfinite(p.grad).all(), name


def test_overfits_single_batch():
    nf.seed_everything(0)
    m = GPT(small_cfg(n_embd=32))
    opt = AdamW(m.parameters(), lr=5e-3, weight_decay=0.0)
    x = np.random.randint(0, 17, (4, 8))
    y = np.roll(x, -1, axis=1)
    first = None
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first


def test_generate_respects_context_and_top_k():
    m = GPT(small_cfg())
    out = m.generate(np.array([[1, 2, 3]]), max_new_tokens=12, top_k=1)
    assert out.shape == (1, 15)
    assert (out[0, :3] == [1, 2, 3]).all()
    assert m.training  # generate restores train mode
    out2 = m.generate(np.array([1, 2, 3]), max_new_tokens=2, temperature=0.5)
    assert out2.shape == (1, 5)


def test_attention_maps_available_after_forward():
    m = GPT(small_cfg())
    m(np.random.randint(0, 17, (1, 6)))
    maps = m.attention_maps()
    assert len(maps) == 2 and maps[0].shape == (1, 2, 6, 6)
    assert np.allclose(maps[0].sum(-1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = GPT(small_cfg())
    opt = AdamW(m.parameters(), lr=1e-3)
    opt.t = 42
    path = os.path.join(tmp_path, "ckpt.npz")
    save_checkpoint(path, m, opt, extra={"note": "hi"})
    meta = load_checkpoint(path)
    assert meta["config"]["vocab_size"] == 17 and meta["extra"]["note"] == "hi"
    m2 = GPT(GPTConfig(**meta["config"]))
    opt2 = AdamW(m2.parameters(), lr=0.0)
    load_checkpoint(path, m2, opt2)
    assert opt2.t == 42 and opt2.lr == 1e-3
    idx = np.random.randint(0, 17, (1, 8))
    assert np.allclose(m(idx).data, m2(idx).data, atol=1e-6)
