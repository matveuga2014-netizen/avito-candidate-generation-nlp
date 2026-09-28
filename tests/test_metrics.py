import numpy as np

from src.metrics import recall_at_k, recall_per_query


def test_half():
    # Аналог примера из задания: 2 релевантных, найдено 1 -> 0.5
    assert recall_at_k({"q": ["a", "x", "y"]}, {"q": {"a", "b"}}) == 0.5


def test_mean_over_queries_and_missing_is_zero():
    truth = {"q1": {"a"}, "q2": {"b"}}
    # q2 без предсказания: 0, а не выпадение из среднего
    assert recall_at_k({"q1": ["a"]}, truth) == 0.5
    assert recall_per_query({"q1": ["a"]}, truth).to_dict() == {"q1": 1.0, "q2": 0.0}


def test_only_top_k_counts():
    pred = {"q": [f"x{i}" for i in range(50)] + ["a"]}
    assert recall_at_k(pred, {"q": {"a"}}) == 0.0


def test_oracle_and_random():
    rng = np.random.default_rng(0)
    corpus = [f"i{i}" for i in range(1000)]
    truth = {f"q{j}": {corpus[j]} for j in range(1000)}
    assert recall_at_k({q: sorted(r) for q, r in truth.items()}, truth) == 1.0
    rand = {q: list(rng.choice(corpus, 50, replace=False)) for q in truth}
    assert abs(recall_at_k(rand, truth) - 50 / 1000) < 0.02


def test_weighted():
    import pandas as pd
    truth = {"q1": {"a"}, "q2": {"b"}}
    pred = {"q1": ["a"], "q2": ["x"]}
    assert recall_at_k(pred, truth, weights=pd.Series({"q1": 3.0, "q2": 1.0})) == 0.75


def test_paired_bootstrap():
    from src.metrics import paired_bootstrap
    a, b = [1.0] * 200, [0.0] * 200
    assert paired_bootstrap(a, b) == (1.0, 1.0, 1.0)
    mean, lo, hi = paired_bootstrap([1, 0] * 100, [0, 1] * 100)
    assert mean == 0.0 and lo < 0 < hi
