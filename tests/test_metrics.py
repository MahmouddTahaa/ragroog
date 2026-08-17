from clinical_rag.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_recall_hit_mrr():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3
    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_ndcg_perfect_and_empty():
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, 2) == 1.0
    assert ndcg_at_k([], relevant, 5) == 0.0
    assert ndcg_at_k(["x", "y"], set(), 5) == 0.0


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([], 95) == 0.0
