"""Cross-encoder reranking. Scores (query, passage) pairs jointly, so it is slower but
more precise than bi-encoder similarity. Used only on a small candidate set."""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from app.config import Settings
from app.models import Hit


class NoOpReranker:
    is_real = False

    def rerank(self, query: str, hits: List[Hit], top_k: Optional[int] = None) -> List[Hit]:
        return hits[:top_k] if top_k else hits


class CrossEncoderReranker:
    is_real = True

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[Hit], top_k: Optional[int] = None) -> List[Hit]:
        if not hits:
            return []
        logits = np.asarray(self.model.predict([(query, h.chunk.text) for h in hits]), dtype="float64")
        probs = 1.0 / (1.0 + np.exp(-logits))     # logits -> probability-like score in [0, 1]
        out = [replace(h, rerank_score=float(p)) for h, p in zip(hits, probs)]
        out.sort(key=lambda h: h.rerank_score, reverse=True)
        return out[:top_k] if top_k else out


def get_reranker(settings: Settings):
    if settings.reranker_backend == "none":
        return NoOpReranker()
    return CrossEncoderReranker(settings.reranker_model)
