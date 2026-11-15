"""
neuroforge.nn.transformer
=========================

A decoder-only GPT built entirely on NeuroForge primitives.

Architecture (per block, pre-LayerNorm as in GPT-2):

    x = x + MultiHeadCausalSelfAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from .. import functional as F
from ..tensor import Tensor, no_grad
from .layers import Dropout, Embedding, LayerNorm, Linear
from .module import Module


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64        # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True    # share token embedding and output projection

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout_p = cfg.dropout
        self.resid_drop = Dropout(cfg.dropout)
        self.last_attention: Optional[np.ndarray] = None  # (B, nh, T, T), for visualisation

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                                # (B, T, 3C)
        q = qkv[:, :, : C].reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = qkv[:, :, C: 2 * C].reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = qkv[:, :, 2 * C:].reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y, att = F.scaled_dot_product_attention(q, k, v, causal=True, dropout_p=self.dropout_p,
                                                training=self.training, return_weights=True)
        self.last_attention = att.data
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(self.fc(x).gelu()))


class TransformerBlock(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = Embedding(cfg.block_size, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)
        self.blocks: List[TransformerBlock] = [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        # GPT-2 style init: scale residual projections by 1/sqrt(2 * n_layer)
        scale = 1.0 / np.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            blk.attn.proj.weight.data *= scale
            blk.mlp.proj.weight.data *= scale

    def forward(self, idx, targets=None):
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(np.int64)
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        pos = np.arange(T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))            # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B, T, V)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def attention_maps(self) -> List[np.ndarray]:
        """Attention weights of every layer from the most recent forward pass."""
        return [blk.attn.last_attention for blk in self.blocks]

    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, stop_token: Optional[int] = None) -> np.ndarray:
        """Autoregressively sample ``max_new_tokens`` tokens after ``idx`` (shape ``(B, T)``)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        was_training = self.training
        self.eval()
        with no_grad():
            for _ in range(max_new_tokens):
                ctx = idx[:, -self.cfg.block_size:]
                logits = self.forward(ctx)[:, -1, :].data / max(temperature, 1e-6)   # (B, V)
                if top_k is not None:
                    kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                    logits = np.where(logits < kth, -np.inf, logits)
                probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True)
                nxt = np.array([np.random.choice(probs.shape[-1], p=p) for p in probs])[:, None]
                idx = np.concatenate([idx, nxt], axis=1)
                if stop_token is not None and (nxt == stop_token).all():
                    break
        self.train(was_training)
        return idx
