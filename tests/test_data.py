"""Tokenisers, datasets and the dashboard server."""

import json
import os
import urllib.request

import numpy as np

from neuroforge.data import BPETokenizer, CharTokenizer, DataLoader, TextDataset, make_arithmetic, make_shapes, make_spiral


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer("hello world")
    assert tok.vocab_size == len(set("hello world"))
    assert tok.decode(tok.encode("hello")) == "hello"
    p = os.path.join(tmp_path, "tok.json")
    tok.save(p)
    assert CharTokenizer.load(p).encode("world") == tok.encode("world")


def test_bpe_tokenizer_trains_merges_and_roundtrips(tmp_path):
    text = "the cat sat on the mat. the cat ate the rat. " * 20
    tok = BPETokenizer().train(text, vocab_size=300)
    assert 256 < tok.vocab_size <= 300
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))  # compression happened
    assert tok.decode(ids) == text
    assert tok.decode(tok.encode("unicode ✓ ok")) == "unicode ✓ ok"
    p = os.path.join(tmp_path, "bpe.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.encode(text) == ids


def test_text_dataset_batches_are_shifted_by_one():
    ds = TextDataset(np.arange(1000), block_size=8)
    x, y = ds.get_batch(4)
    assert x.shape == (4, 8) and np.array_equal(x + 1, y)
    xv, _ = ds.get_batch(2, "val")
    assert xv.min() >= 900


def test_dataloader_covers_every_sample():
    X, y = np.arange(23)[:, None], np.arange(23)
    seen = np.concatenate([yb for _, yb in DataLoader(X, y, batch_size=5)])
    assert sorted(seen.tolist()) == list(range(23))
    assert len(DataLoader(X, y, batch_size=5)) == 5


def test_synthetic_datasets():
    X, y = make_spiral(50, 3)
    assert X.shape == (150, 2) and set(y.tolist()) == {0, 1, 2}
    X, y = make_shapes(20, size=16)
    assert X.shape == (20, 1, 16, 16) and X.min() >= 0 and X.max() <= 1
    lines = make_arithmetic(5, n_digits=2)
    for line in lines:
        a, rest = line.split("+")
        b, c = rest.rstrip(";").split("=")
        assert int(a) + int(b) == int(c) and len(line) == len(lines[0])


def test_dashboard_serves_state_and_generation():
    from neuroforge.dashboard import Dashboard

    dash = Dashboard(port=0)  # port 0 -> OS picks a free port
    port = dash._server.server_address[1]
    dash.generate_fn = lambda prompt, n, temp: prompt + "!" * n
    dash.set_config(model="test")
    dash.log(1, loss=1.5, lr=1e-3)
    dash.add_sample(1, "hello")
    try:
        state = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state"))
        assert state["metrics"][0]["loss"] == 1.5 and state["config"]["model"] == "test"
        assert state["samples"][0]["text"] == "hello" and state["has_generate"]
        gen = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/generate?prompt=hi&tokens=3"))
        assert gen["text"] == "hi!!!"
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "NeuroForge" in html
    finally:
        dash.close()
