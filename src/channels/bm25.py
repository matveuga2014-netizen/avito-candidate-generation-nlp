"""BM25 (bm25s) отдельно по заголовку, параметрам и началу описания.

Отдельные индексы, а не один склеенный: у полей разная длина и шум
(параметры длинные и шаблонные), ранкер/RRF сам решит, чему верить.
Стоп-слова не удаляем: «без», «на» в услугах несут смысл.
"""
import bm25s
import numpy as np
import pandas as pd
import Stemmer

from src.data import norm_text, query_text

STEM = Stemmer.Stemmer("russian")
FIELDS = {
    "title": lambda c: c["item_title_raw"],
    "params": lambda c: c["item_infm_params_text"],
    "desc": lambda c: c["item_description_raw"].fillna("").str.slice(0, 1000),
}


def _tok(texts: pd.Series):
    return bm25s.tokenize(norm_text(texts).tolist(), stopwords=None, stemmer=STEM,
                          return_ids=False, show_progress=False)


def bm25_channels(queries: pd.DataFrame, corpus: pd.DataFrame, k: int = 500) -> dict[str, pd.DataFrame]:
    """{bm25_<поле>: DataFrame(qid, item_id, score, rank)}; только документы со score > 0."""
    qtok = _tok(query_text(queries))
    ids = corpus["item_id"].to_numpy()
    out = {}
    for name, get in FIELDS.items():
        r = bm25s.BM25()
        r.index(_tok(get(corpus)), show_progress=False)
        docs, scores = r.retrieve(qtok, k=k, show_progress=False, n_threads=-1)
        n = docs.shape[1]
        df = pd.DataFrame({
            "qid": np.repeat(queries["qid"].to_numpy(), n),
            "item_id": ids[docs.ravel()],
            "score": scores.ravel(),
        })
        df = df[df["score"] > 0]
        # стабильный порядок: score убыв., при равенстве item_id
        df = df.sort_values(["qid", "score", "item_id"], ascending=[True, False, True], kind="stable")
        df["rank"] = df.groupby("qid").cumcount() + 1
        out[f"bm25_{name}"] = df.reset_index(drop=True)
    return out


def bm25_pair_scores(queries: pd.DataFrame, corpus: pd.DataFrame, pairs: pd.DataFrame, field: str = "title") -> np.ndarray:
    """BM25 по полю для заданных пар (qid, item_id), в том числе вне топа канала.

    Нужен ранкеру: у кандидатов из priors/pop_loc нет ранга BM25, и ранкер не
    видит, что объявление не про запрос (8% промахов P3 — местные нерелевантные).
    """
    r = bm25s.BM25()
    r.index(_tok(FIELDS[field](corpus)), show_progress=False)
    qtok = dict(zip(queries["qid"], _tok(query_text(queries))))
    col = pd.Series(np.arange(len(corpus)), index=corpus["item_id"])
    out = np.zeros(len(pairs), dtype=np.float32)
    for qid, idx in pairs.groupby("qid").indices.items():
        if not qtok[qid]:  # запрос без слов из 2+ символов: сходство 0, bm25s пустой запрос не принимает
            continue
        out[idx] = r.get_scores(qtok[qid])[col[pairs["item_id"].to_numpy()[idx]].to_numpy()]
    return out
