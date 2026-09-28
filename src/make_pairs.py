"""Пары «запрос -> выбранное объявление» для дообучения dense-модели на Kaggle.

Только история для валидации (eval_history: P1 с прореживанием), чтобы P2/P3
остались честными. Тексты строят те же функции, что и dense-канал на
инференсе (query_text, item_dense_text) с префиксами E5.

Запуск: python -m src.make_pairs -> artifacts/kaggle/pairs_p1.parquet
                                  и artifacts/kaggle/items_text.parquet
items_text: тексты локального корпуса (он включает весь benchmark_items) без
префиксов, тем же item_dense_text, что локально: ноутбук кодирует их на Kaggle.
"""
import numpy as np
import pandas as pd

from src.data import item_dense_text, load, query_text
from src.split import ARTIFACTS, SEED, eval_history, load_split, local_corpus

# Потолки против перекоса: без них модель учит популярные объявления
# и частые запросы («маникюр» — 6.5k экземпляров), а не соответствие текста.
MAX_PER_ITEM = 10
MAX_PER_ANCHOR = 50


def write_items_text() -> None:
    corpus = local_corpus(load_split())
    out = ARTIFACTS / "kaggle" / "items_text.parquet"
    out.parent.mkdir(exist_ok=True)
    df = pd.DataFrame({"item_id": corpus["item_id"], "text": item_dense_text(corpus)})
    assert df["item_id"].is_unique and df["text"].str.len().gt(0).all()
    df.to_parquet(out, index=False)
    print(f"записан {out}: {len(df)} объявлений ({out.stat().st_size / 1e6:.0f} МБ)")


def main() -> None:
    write_items_text()
    hist = eval_history(load_split())
    pairs = hist.assign(anchor="query: " + query_text(hist))[["anchor", "items"]].explode("items")
    pairs = pairs.rename(columns={"items": "item_id"}).drop_duplicates()
    n_unique = len(pairs)

    rng = np.random.default_rng(SEED)
    pairs = pairs.sort_values(["anchor", "item_id"]).assign(r=rng.random(len(pairs)))
    pairs = pairs.sort_values("r")
    pairs = pairs[pairs.groupby("item_id").cumcount() < MAX_PER_ITEM]
    pairs = pairs[pairs.groupby("anchor").cumcount() < MAX_PER_ANCHOR]

    cols = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
    items = load("train", cols, filters=[("item_id", "in", sorted(set(pairs["item_id"])))]).drop_duplicates("item_id")
    items["positive"] = "passage: " + item_dense_text(items)
    pairs = pairs.merge(items[["item_id", "positive"]], on="item_id", how="inner")
    assert pairs["positive"].notna().all()
    assert not pairs["anchor"].str.contains(r"\b(?:nan|none)\b", regex=True).any()

    # Порядок фиксирован seed, trainer всё равно перемешивает со своим seed
    pairs = pairs.sort_values("r")[["anchor", "positive"]].reset_index(drop=True)
    out = ARTIFACTS / "kaggle" / "pairs_p1.parquet"
    out.parent.mkdir(exist_ok=True)
    pairs.to_parquet(out, index=False)
    print(f"уникальных пар {n_unique}, после потолков {len(pairs)}, анкеров {pairs['anchor'].nunique()}")
    print(f"записан {out} ({out.stat().st_size / 1e6:.0f} МБ)")
    print(pairs.head(3).to_string(max_colwidth=90))


if __name__ == "__main__":
    main()
