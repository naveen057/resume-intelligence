"""Embedding backends behind one tiny interface."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from app.config import Settings
from app.text import tokenize


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic signed feature-hashing of unigrams + bigrams.

    Not semantic. It exists so tests, CI and offline demos run with zero downloads.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            toks = tokenize(text)
            for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(feat.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)
        return vecs.astype("float32")


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
