"""Бейзлайн «популярное в локации» и контрольные цифры метрики (оракул, случайный ответ).

Запуск: python -m src.baseline
Печатает Recall@50 на P3 для случайного ответа, оракула и бейзлайна,
и пишет artifacts/baseline_answer.csv для benchmark той же функцией.
"""
import numpy as np
import pandas as pd

from src.data import load
from src.metrics import K, recall_at_k
from src.split import ARTIFACTS, SEED, eval_history, load_split, local_corpus, location_weights


def popular_in_location(queries: pd.DataFrame, history: pd.DataFrame, corpus: pd.DataFrame, k: int = K) -> dict[str, list[str]]:
    """Топ-k объявлений корпуса в локации запроса по популярности в истории.

    Популярность = число экземпляров истории, где объявление выбрали. Большинство
    объявлений корпуса в истории не встречается, поэтому вторым ключом идёт
    число отзывов, третьим item_id для стабильного порядка. Недобор в локации
    добивается глобальным топом, чтобы всегда было ровно k.
    """
    pop = history["items"].explode().value_counts()
    c = corpus[["item_id", "item_location_id", "item_rating_reviews_count"]].copy()
    c["pop"] = c["item_id"].map(pop).fillna(0)
    c["rev"] = c["item_rating_reviews_count"].fillna(0)
    c = c.sort_values(["pop", "rev", "item_id"], ascending=[False, False, True], kind="stable")
    global_top = c["item_id"].head(2 * k).tolist()
    by_loc = c.groupby("item_location_id", sort=False)["item_id"].agg(lambda s: s.head(k).tolist())

    out = {}
    for qid, loc in zip(queries["qid"], queries["search_location_id"]):
        lst = list(by_loc.get(loc, []))
        seen = set(lst)
        for i in global_top:
            if len(lst) == k:
                break
            if i not in seen:
                lst.append(i)
                seen.add(i)
        out[qid] = lst
    return out


def main() -> None:
    sp = load_split()
    p3 = sp[sp["part"] == "P3"]
    truth = {q: set(it) for q, it in zip(p3["qid"], p3["items"])}
    corpus = local_corpus(sp)
    ids = corpus["item_id"].to_numpy()

    # Один прогон случайного ответа на 5000 запросах даёт ~1 попадание, поэтому
    # усредняем 20 прогонов. Повторы id внутри 50 (вероятность ~0.6%) на проверку не влияют.
    rng = np.random.default_rng(SEED)
    rand = np.mean([recall_at_k({q: ids[rng.integers(0, len(ids), K)].tolist() for q in truth}, truth) for _ in range(20)])
    print(f"случайные 50, среднее 20 прогонов: {rand:.5f} (ожидание 50/|корпус| = {K / len(ids):.5f})")
    print(f"оракул: {recall_at_k({q: sorted(r) for q, r in truth.items()}, truth):.4f}")

    pred = popular_in_location(p3, eval_history(sp), corpus)
    assert all(len(v) == K == len(set(v)) for v in pred.values())
    w = pd.Series(location_weights(p3["search_location_id"]).to_numpy(), index=p3["qid"])
    print(f"популярное в локации, P3: {recall_at_k(pred, truth):.4f}, взвешенно {recall_at_k(pred, truth, weights=w):.4f}")

    # Та же функция на benchmark: история = весь train, корпус = benchmark_items
    bq = load("benchmark_queries").rename(columns={"query_id": "qid"})
    ans = popular_in_location(bq, sp, load("benchmark_items"))
    out = ARTIFACTS / "baseline_answer.csv"
    pd.DataFrame({"query_id": list(ans), "answer": [" ".join(v) for v in ans.values()]}).to_csv(out, index=False)
    print("записан", out)


if __name__ == "__main__":
    main()
