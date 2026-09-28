"""Recall@K как на платформе: среднее по запросам |топ-K ∩ релевантные| / |релевантные|."""
import pandas as pd

K = 50


def recall_per_query(pred: dict[str, list[str]], truth: dict[str, set[str]], k: int = K) -> pd.Series:
    """Recall каждого запроса из truth.

    Запрос без предсказания получает 0, а не выпадает из среднего:
    иначе пропущенные запросы завышали бы цифру.
    """
    out = {}
    for qid, rel in truth.items():
        assert rel, f"у запроса {qid} нет релевантных"
        top = set(pred.get(qid, [])[:k])
        out[qid] = len(top & rel) / len(rel)
    return pd.Series(out, dtype=float)


def recall_at_k(pred: dict[str, list[str]], truth: dict[str, set[str]], k: int = K,
                weights: pd.Series | None = None) -> float:
    """Средний recall; weights (индекс = qid) — веса по локации для P2/P3."""
    r = recall_per_query(pred, truth, k)
    if weights is None:
        return float(r.mean())
    w = weights.reindex(r.index)
    assert w.notna().all(), "нет веса для части запросов"
    return float((r * w).sum() / w.sum())


def paired_bootstrap(a, b, w=None, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Разница взвешенных средних a - b по запросам и её 95% интервал (парный бутстрэп).

    a, b — recall по одним и тем же запросам в одном порядке; w — веса запросов.
    """
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float((d * w).sum() / w.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
