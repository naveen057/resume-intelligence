"""Small, dependency-free BM25 (Okapi). Good at exact skill terms like 'FastAPI' or 'C++'."""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

from app.text import tokenize


class BM25Index:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.tf = [Counter(t) for t in self.doc_tokens]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if self.n and self.doc_len.sum() else 1.0
        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype="float32")
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
