"""Каналы из истории: похожие запросы -> выбранные объявления (logs),
подкатегории похожих запросов -> объявления корпуса в локации (priors),
популярное в локации (pop_loc, как бейзлайн).

Всё строится только по переданной истории: для P2/P3 это eval_history,
для benchmark весь train. TF-IDF обучается на текстах истории, не запросов.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baseline import popular_in_location
from src.data import load

N_NEIGHBORS = 30
MIN_SIM = 0.3
LOC_BONUS = 1.0  # выбор в той же локации весит вдвое: 83% выборов в локации запроса


def neighbors(queries: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Похожие тексты истории: qid, htext, sim. Точный текст сюда попадает с sim=1.

    char_wb 3–5-граммы ловят опечатки и словоформы без лемматизации.
    """
    htexts = np.sort(history["qtext"].unique())
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32, sublinear_tf=True)
    H = vec.fit_transform(htexts)
    Q = vec.transform(queries["qtext"])
    out = []
    for s in range(0, Q.shape[0], 256):
        sim = (Q[s:s + 256] @ H.T).toarray()
        top = np.argpartition(-sim, N_NEIGHBORS, axis=1)[:, :N_NEIGHBORS]
        for j, row in enumerate(top):
            sc = sim[j, row]
            keep = sc >= MIN_SIM
            out.append(pd.DataFrame({"qid": queries["qid"].iat[s + j], "htext": htexts[row[keep]], "sim": sc[keep]}))
    return pd.concat(out, ignore_index=True)


def microcat_probs(nb: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """P(microcat | запрос): выборы похожих текстов истории, взвешенные по сходству.

    Возвращает qid, mc, p; внутри запроса по убыванию p (при равенстве по mc).
    Общая для канала priors и признака ранкера, чтобы расчёт совпадал.
    """
    mc = load("train", ["item_id", "item_microcat_id"]).drop_duplicates("item_id").set_index("item_id")["item_microcat_id"]
    ex = history[["qtext", "items"]].explode("items")
    ex["mc"] = ex["items"].map(mc)
    exm = ex.groupby(["qtext", "mc"]).size().rename("cnt").reset_index()
    pm = nb.merge(exm, left_on="htext", right_on="qtext").assign(w=lambda d: d["sim"] * d["cnt"])
    pm = pm.groupby(["qid", "mc"], as_index=False)["w"].sum()
    pm["p"] = pm["w"] / pm.groupby("qid")["w"].transform("sum")
    return pm.sort_values(["qid", "p", "mc"], ascending=[True, False, True])[["qid", "mc", "p"]].reset_index(drop=True)


def _rank(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.sort_values(["qid", "score", "item_id"], ascending=[True, False, True], kind="stable")
    df["rank"] = df.groupby("qid").cumcount() + 1
    return df[df["rank"] <= k].reset_index(drop=True)


def history_channels(queries: pd.DataFrame, history: pd.DataFrame, corpus: pd.DataFrame, k: int = 500,
                     use_popularity: bool = True) -> dict[str, pd.DataFrame]:
    """use_popularity=False — только для абляции: внутри ячейки priors сортировка по
    рейтингу и отзывам вместо популярности в истории."""
    nb = neighbors(queries, history)
    qloc = queries.set_index("qid")["search_location_id"]
    item_loc = corpus.set_index("item_id")["item_location_id"]

    # Выборы истории: (текст, объявление) -> число экземпляров
    ex = history[["qtext", "items"]].explode("items").rename(columns={"items": "item_id"})
    ex = ex.groupby(["qtext", "item_id"]).size().rename("cnt").reset_index()

    # logs: sum sim * cnt по похожим текстам, только объявления корпуса, бонус за локацию
    m = nb.merge(ex, left_on="htext", right_on="qtext")
    m = m[m["item_id"].isin(item_loc.index)]
    same_loc = m["item_id"].map(item_loc).to_numpy() == m["qid"].map(qloc).to_numpy()
    m["score"] = m["sim"] * np.log1p(m["cnt"]) * (1 + LOC_BONUS * same_loc)
    logs = _rank(m.groupby(["qid", "item_id"], as_index=False)["score"].sum(), k)

    # priors: P(microcat | запрос) по похожим текстам -> объявления корпуса этих microcat в локации
    pm = microcat_probs(nb, history).groupby("qid").head(3)

    pop = history["items"].explode().value_counts()
    c = corpus[["item_id", "item_location_id", "item_microcat_id", "item_rating_reviews_count"]].copy()
    # внутри подкатегории: популярность в истории, потом отзывы (как в бейзлайне)
    if use_popularity:
        c["within"] = 0.01 * np.log1p(c["item_id"].map(pop).fillna(0)) + 0.001 * np.log1p(c["item_rating_reviews_count"].fillna(0))
    else:
        rating = corpus["item_rating"].fillna(0).to_numpy()
        c["within"] = 0.01 * rating / 5 + 0.001 * np.log1p(c["item_rating_reviews_count"].fillna(0))
    pq_ = pm.merge(queries[["qid", "search_location_id"]], on="qid")
    pr = pq_.merge(c, left_on=["mc", "search_location_id"], right_on=["item_microcat_id", "item_location_id"])
    pr["score"] = pr["p"] + pr["within"]
    priors = _rank(pr[["qid", "item_id", "score"]], k)

    # pop_loc: бейзлайн как канал-фолбэк (всегда ровно 50 на запрос)
    pl = popular_in_location(queries, history, corpus)
    pop_loc = pd.DataFrame([(q, i, 50 - r, r + 1) for q, lst in pl.items() for r, i in enumerate(lst)],
                           columns=["qid", "item_id", "score", "rank"])
    return {"logs": logs, "priors": priors, "pop_loc": pop_loc}
