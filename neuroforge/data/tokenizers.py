"""
neuroforge.data.tokenizers
==========================

Two tokenisers:

* ``CharTokenizer`` - one token per unique character (great for small corpora).
* ``BPETokenizer``  - byte-level Byte Pair Encoding, the algorithm used by
  GPT-2/3/4. Trained from scratch on any text; no external dependencies.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


class CharTokenizer:
    def __init__(self, text: Optional[str] = None, chars: Optional[List[str]] = None):
        if chars is None:
            assert text is not None, "provide text or chars"
            chars = sorted(set(text))
        self.chars = list(chars)
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(chars=json.load(f)["chars"])


class BPETokenizer:
    """Byte-level BPE.

    Training repeatedly merges the most frequent adjacent pair of tokens until
    the vocabulary reaches ``vocab_size``. The first 256 tokens are raw bytes,
    so *any* string can be encoded.
    """

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pair_counts(ids: List[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i, n = [], 0, len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i % 50 == 0 or i == vocab_size - 257):
                print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {new_id} ({self.vocab[new_id]!r}) x{freq}")
        return self

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # pick the pair that was merged earliest during training
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "merges": [[a, b, c] for (a, b), c in self.merges.items()]}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for a, b, c in data["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        return tok
