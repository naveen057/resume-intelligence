"""FAISS inner-product index (vectors are L2-normalised, so IP == cosine). Falls back to numpy."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
        except ImportError:
            self._index = None
            self._mat = np.zeros((0, dim), dtype="float32")
            self.backend = "numpy"

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._mat = np.vstack([self._mat, vectors])

    def __len__(self) -> int:
        return int(self._index.ntotal) if self._index is not None else len(self._mat)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        k = min(k, len(self))
        if k == 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = (self._mat @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
