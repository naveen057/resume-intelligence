import numpy as np

from app.ingestion.chunker import chunk_text
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.hybrid import HybridRetriever, rrf
from app.retrieval.vector_store import VectorStore
from app.text import tokenize
from tests.helpers import RESUME


def test_tokenizer_keeps_technical_terms():
    assert tokenize("C++ and scikit-learn, CI/CD.") == ["c++", "and", "scikit-learn", "ci", "cd"]


def test_bm25_ranks_exact_term_first():
    idx = BM25Index(["python pandas", "java spring", "fastapi docker python"])
    assert idx.top_k("fastapi", 3)[0][0] == 2
    assert idx.top_k("rust", 3) == []


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] > fused[3] and fused[2] > fused[4]
    assert abs(fused[1] - (1 / 61 + 1 / 62)) < 1e-9


def test_vector_store_returns_self_as_nearest():
    emb = HashingEmbedder()
    vecs = emb.encode(["alpha beta", "gamma delta", "epsilon zeta"])
    store = VectorStore(emb.dim)
    store.add(vecs)
    ids = [i for i, _ in store.search(vecs[1], 2)]
    assert ids[0] == 1 and len(store) == 3


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = emb.encode(["python sql"]), emb.encode(["python sql"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def test_hybrid_finds_fastapi_chunk_in_every_mode():
    retriever = HybridRetriever(chunk_text(RESUME), HashingEmbedder())
    for mode in ("bm25", "dense", "hybrid"):
        top = retriever.search("FastAPI", k=3, mode=mode)
        assert any("FastAPI" in h.chunk.text for h in top), mode


def test_empty_corpus_and_bad_mode():
    r = HybridRetriever([], HashingEmbedder())
    assert r.search("x") == []
    try:
        HybridRetriever(chunk_text(RESUME), HashingEmbedder()).search("x", mode="nope")
        raise AssertionError
    except ValueError:
        pass
