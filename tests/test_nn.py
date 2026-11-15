"""Layers, functional ops, and the conv/pool/attention backward passes."""

import numpy as np
import pytest

import neuroforge as nf
import neuroforge.functional as F
from neuroforge import Tensor, nn
from neuroforge.utils import gradcheck


def test_linear_shapes_and_grad():
    layer = nn.Linear(5, 3)
    x = nf.randn(7, 5, requires_grad=True)
    y = layer(x)
    assert y.shape == (7, 3)
    y.sum().backward()
    assert layer.weight.grad.shape == (3, 5)
    assert layer.bias.grad.shape == (3,)
    assert x.grad.shape == (7, 5)


def test_linear_matches_numpy():
    layer = nn.Linear(4, 2)
    x = nf.randn(3, 4)
    expected = x.data @ layer.weight.data.T + layer.bias.data
    assert np.allclose(layer(x).data, expected, atol=1e-6)


def test_embedding_gradient_is_scatter_add():
    emb = nn.Embedding(10, 4)
    idx = np.array([[1, 1, 3]])
    out = emb(idx)
    assert out.shape == (1, 3, 4)
    out.sum().backward()
    assert np.allclose(emb.weight.grad[1], 2.0)  # used twice
    assert np.allclose(emb.weight.grad[3], 1.0)
    assert np.allclose(emb.weight.grad[0], 0.0)


def test_layernorm_normalises():
    ln = nn.LayerNorm(8)
    x = nf.randn(4, 8) * 5 + 3
    y = ln(x).data
    assert np.allclose(y.mean(-1), 0, atol=1e-5)
    assert np.allclose(y.std(-1), 1, atol=1e-3)


def test_layernorm_gradcheck():
    w, b = Tensor(np.random.randn(6)), Tensor(np.random.randn(6))
    x = Tensor(np.random.randn(3, 6))
    assert gradcheck(lambda a, ww, bb: (F.layer_norm(a, ww, bb) * Tensor(np.arange(18.).reshape(3, 6))).sum(), x, w, b)


def test_dropout_train_vs_eval():
    d = nn.Dropout(0.5)
    x = nf.ones(1000)
    y = d(x)
    kept = (y.data != 0).mean()
    assert 0.35 < kept < 0.65
    assert np.allclose(y.data[y.data != 0], 2.0)  # inverted scaling
    d.eval()
    assert np.allclose(d(x).data, 1.0)


def test_conv2d_matches_naive_loop():
    x = nf.randn(2, 3, 5, 5)
    w = nf.randn(4, 3, 3, 3)
    b = nf.randn(4)
    out = F.conv2d(x, w, b, stride=1, padding=1).data
    xp = np.pad(x.data, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros_like(out)
    for n in range(2):
        for f in range(4):
            for i in range(5):
                for j in range(5):
                    ref[n, f, i, j] = (xp[n, :, i:i + 3, j:j + 3] * w.data[f]).sum() + b.data[f]
    assert np.allclose(out, ref, atol=1e-4)


def test_conv2d_stride_and_gradcheck():
    x, w, b = Tensor(np.random.randn(1, 2, 6, 6)), Tensor(np.random.randn(3, 2, 3, 3)), Tensor(np.random.randn(3))
    out = F.conv2d(x, w, b, stride=2, padding=1)
    assert out.shape == (1, 3, 3, 3)
    assert gradcheck(lambda a, ww, bb: (F.conv2d(a, ww, bb, stride=2, padding=1) ** 2).sum(), x, w, b)


def test_maxpool_gradcheck():
    x = Tensor(np.random.randn(2, 2, 6, 6))
    y = F.max_pool2d(x, 2)
    assert y.shape == (2, 2, 3, 3)
    assert gradcheck(lambda a: (F.max_pool2d(a, 2) ** 2).sum(), x)
    assert gradcheck(lambda a: (F.max_pool2d(a, 3, stride=1) ** 2).sum(), x)


def test_avgpool():
    x = nf.ones(1, 1, 4, 4)
    assert np.allclose(F.avg_pool2d(x, 2).data, 1.0)


def test_cross_entropy_matches_manual():
    logits = nf.randn(5, 7, requires_grad=True)
    targets = np.array([0, 3, 6, 1, 1])
    loss = F.cross_entropy(logits, targets)
    p = np.exp(logits.data - logits.data.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    manual = -np.log(p[np.arange(5), targets]).mean()
    assert np.isclose(loss.item(), manual, atol=1e-5)
    loss.backward()
    expected_grad = p.copy()
    expected_grad[np.arange(5), targets] -= 1
    assert np.allclose(logits.grad, expected_grad / 5, atol=1e-5)


def test_cross_entropy_ignore_index_and_smoothing():
    logits = Tensor(np.random.randn(4, 3))
    t = np.array([0, 1, -1, 2])
    loss = F.cross_entropy(logits, t, ignore_index=-1)
    loss_full = F.cross_entropy(logits[np.array([0, 1, 3])], np.array([0, 1, 2]))
    assert np.isclose(loss.item(), loss_full.item(), atol=1e-6)
    assert gradcheck(lambda a: F.cross_entropy(a, t, ignore_index=-1, label_smoothing=0.1), logits)


def test_bce_and_mse_gradcheck():
    x = Tensor(np.random.randn(3, 2))
    y = np.random.rand(3, 2).astype(np.float32)
    assert gradcheck(lambda a: F.binary_cross_entropy_with_logits(a, y), x)
    assert gradcheck(lambda a: F.mse_loss(a, y), x)
    assert gradcheck(lambda a: F.huber_loss(a * 3, y), x)


def test_attention_is_causal():
    q = nf.randn(1, 2, 5, 4)
    k = nf.randn(1, 2, 5, 4)
    v = nf.randn(1, 2, 5, 4)
    _, att = F.scaled_dot_product_attention(q, k, v, causal=True, return_weights=True)
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.allclose(att.data[..., upper], 0.0)
    assert np.allclose(att.data.sum(-1), 1.0, atol=1e-5)


def test_attention_gradcheck():
    q, k, v = (Tensor(np.random.randn(1, 2, 4, 3)) for _ in range(3))
    assert gradcheck(lambda a, b, c: (F.scaled_dot_product_attention(a, b, c, causal=True) ** 2).sum(), q, k, v)


def test_module_parameter_discovery_and_state_dict():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 3)
            self.blocks = [nn.Linear(3, 3), nn.Linear(3, 1)]
            self.extra = {"ln": nn.LayerNorm(3)}

    net = Net()
    names = [n for n, _ in net.named_parameters()]
    assert "a.weight" in names and "blocks.1.bias" in names and "extra.ln.weight" in names
    assert net.num_parameters() == (2 * 3 + 3) + (3 * 3 + 3) + (3 + 1) + 6
    sd = net.state_dict()
    net2 = Net()
    net2.load_state_dict(sd)
    for (_, p1), (_, p2) in zip(net.named_parameters(), net2.named_parameters()):
        assert np.array_equal(p1.data, p2.data)
    with pytest.raises(KeyError):
        net2.load_state_dict({"a.weight": sd["a.weight"]})


def test_sequential_and_mlp_train_eval_propagate():
    m = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4, 1))
    m.eval()
    assert all(not mod.training for mod in m.modules())
    mlp = nn.MLP([2, 8, 8, 3], dropout=0.1)
    assert mlp(nf.randn(5, 2)).shape == (5, 3)


def test_batchnorm_running_stats():
    bn = nn.BatchNorm1d(3)
    x = nf.randn(64, 3) * 4 + 2
    y = bn(x).data
    assert np.allclose(y.mean(0), 0, atol=1e-4)
    bn.eval()
    assert bn(x).shape == (64, 3)
    assert not np.allclose(bn.running_mean, 0)
