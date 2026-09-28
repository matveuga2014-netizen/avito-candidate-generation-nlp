"""Проверка answer.csv перед отправкой: число строк, id, ровно 50 уникальных объявлений из корпуса.

Запуск: python -m src.validate_answer [путь, по умолчанию answer.csv]
"""
import sys

import pandas as pd

from src.data import DATA_DIR, load

K = 50


def validate(path, query_ids: set[str], corpus_ids: set[str]) -> None:
    """Падает с AssertionError и понятным сообщением на первой проблеме."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    assert header == "query_id,answer", f"заголовок {header!r}: нужны ровно две колонки без индекса"

    # dtype=str и без NaN-подстановки: id должны остаться строками как есть
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(df) == len(query_ids), f"строк {len(df)}, нужно {len(query_ids)}"
    assert df["query_id"].is_unique, "query_id повторяются"
    assert set(df["query_id"]) == query_ids, "набор query_id не совпадает с benchmark"

    for qid, ans in zip(df["query_id"], df["answer"]):
        ids = ans.split(" ")  # split(" "), а не split(): двойной пробел даст пустой id и упадёт ниже
        assert len(ids) == K, f"{qid}: {len(ids)} id вместо {K}"
        assert len(set(ids)) == K, f"{qid}: есть повторы id"
        missing = set(ids) - corpus_ids
        assert not missing, f"{qid}: id не из корпуса, например {sorted(missing)[:3]}"


def main(path: str = "answer.csv") -> None:
    q = set(load("benchmark_queries", ["query_id"])["query_id"])
    items = set(load("benchmark_items", ["item_id"])["item_id"])
    validate(path, q, items)
    print(f"{path}: OK ({len(q)} запросов по {K} id)")


if __name__ == "__main__":
    main(*sys.argv[1:])
