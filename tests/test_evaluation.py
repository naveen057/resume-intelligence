from app.config import ROOT
from app.evaluation.metrics import evaluate_retrieval, recall_at_k, reciprocal_rank
from app.retrieval.embedder import HashingEmbedder


def test_recall_at_k():
    assert recall_at_k([False, True, False], 1, 2) == 0.0
    assert recall_at_k([False, True, True], 3, 2) == 1.0
    assert recall_at_k([True, False, False], 3, 2) == 0.5
    assert recall_at_k([], 3, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False]) == 0.0


def test_eval_dataset_labels_are_valid_and_metrics_bounded():
    res = evaluate_retrieval(ROOT / "data/eval/retrieval_eval.json", HashingEmbedder())
    assert set(res) == {"bm25", "dense", "hybrid"}
    for row in res.values():
        assert all(0.0 <= v <= 1.0 for v in row.values())
        assert row["recall@1"] <= row["recall@3"] <= row["recall@5"]
