"""Retrieval metrics. Relevance is labelled by substring so labels survive re-chunking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from app.config import ROOT
from app.ingestion.chunker import chunk_text
from app.models import Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import MODES, HybridRetriever


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """Fraction of all relevant chunks found in the top k."""
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / min(total_relevant, 10**9)


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def is_relevant(chunk: Chunk, substrings: Sequence[str]) -> bool:
    text = chunk.text.lower()
    return any(s.lower() in text for s in substrings)


def evaluate_retrieval(dataset_path: Path, embedder: Embedder, ks: Sequence[int] = (1, 3, 5),
                       modes: Sequence[str] = MODES) -> Dict[str, Dict[str, float]]:
    data = json.loads(Path(dataset_path).read_text())
    acc: Dict[str, Dict[str, List[float]]] = {m: {f"recall@{k}": [] for k in ks} | {"mrr": []} for m in modes}
    for case in data["cases"]:
        chunks = chunk_text((ROOT / case["resume_path"]).read_text(), source=case["id"])
        retriever = HybridRetriever(chunks, embedder)
        for q in case["queries"]:
            total_rel = sum(is_relevant(c, q["relevant_substrings"]) for c in chunks)
            if total_rel == 0:
                raise ValueError(f"No relevant chunk for query '{q['query']}': fix the label")
            for mode in modes:
                hits = retriever.search(q["query"], k=max(ks), mode=mode)
                rel = [is_relevant(h.chunk, q["relevant_substrings"]) for h in hits]
                for k in ks:
                    acc[mode][f"recall@{k}"].append(recall_at_k(rel, k, total_rel))
                acc[mode]["mrr"].append(reciprocal_rank(rel))
    return {m: {name: round(sum(v) / len(v), 3) for name, v in metrics.items()} for m, metrics in acc.items()}
