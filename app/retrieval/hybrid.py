"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.models import Chunk, Hit
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore

MODES = ("bm25", "dense", "hybrid")


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    """score(d) = sum over rankers of 1 / (k + rank(d)), rank starting at 1."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.text for c in chunks]
        self.bm25 = BM25Index(texts)
        self.store = VectorStore(embedder.dim)
        if chunks:
            self.store.add(embedder.encode(texts))

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               dense_query: Optional[str] = None) -> List[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.chunks:
            return []
        pool = len(self.chunks)
        lex = self.bm25.top_k(query, pool) if mode in {"bm25", "hybrid"} else []
        dense = []
        if mode in {"dense", "hybrid"}:
            qvec = self.embedder.encode([dense_query or query])[0]
            dense = self.store.search(qvec, pool)

        if mode == "bm25":
            scored = dict(lex)
        elif mode == "dense":
            scored = dict(dense)
        else:
            scored = rrf([[i for i, _ in lex], [i for i, _ in dense]])
        order = sorted(scored, key=scored.get, reverse=True)[:k]
        return [Hit(chunk=self.chunks[i], score=float(scored[i])) for i in order]
