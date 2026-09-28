"""CatBoost-ранкер поверх пула каналов (обучение на P2, проверка на P3).

Пул = объединение топ-K_RANKER каналов (K=200: потолок пула 0.913 на P2,
дальше рост медленнее, а строк на 40% больше). Признаки для P2/P3 считаются
по eval_history, для benchmark по всему train — одной функцией build_features.
Группы (запросы) взвешены по локации, как и метрика.

Запуск: python -m src.ranker
"""
import json

import numpy as np
import pandas as pd

from src.candidates import K_RRF, N_ANSWER, cached_candidates, extra_channel
from src.channels.bm25 import bm25_pair_scores
from src.channels.dense import MODEL, pair_cosine
from src.channels.logs import microcat_probs, neighbors
from src.data import haversine_km, location_centroids, norm_text
from src.metrics import paired_bootstrap, recall_at_k, recall_per_query
from src.split import ARTIFACTS, SEED, eval_history, load_split, local_corpus, location_weights

K_RANKER = 200
CHANNELS = ["bm25_desc", "bm25_params", "bm25_title", "dense", "logs", "pop_loc", "priors"]
# Исправления по анализу ошибок (README, «Типы ошибок»): канал dense в радиусе
# от локации запроса и текстовое сходство для всех кандидатов пула.
FIX_CHANNELS = ["dense_local"]
FIX_FEATURES = ["cos_dense", "bm25_title_all"]
# Размера ячейки microcat × локация среди признаков нет: он был первым по важности,
# без него качество P3 не падает, а на benchmark его распределение другое
# (медиана 27 против 8 на P3), поэтому это прямой путь к оптимизму P3.
PRIOR_FEATS = ["rank_priors", "score_priors", "p_mc", "mc_rank"]
NEG_TOP, NEG_RANDOM = 100, 100  # прореживание негативов для обучения (экономия памяти)


def _item_features(corpus: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": corpus["item_id"],
        "item_location_id": corpus["item_location_id"],
        "item_microcat_id": corpus["item_microcat_id"],
        "item_latitude": corpus["item_latitude"].astype(np.float32),
        "item_longitude": corpus["item_longitude"].astype(np.float32),
        "rating": corpus["item_rating"].astype(np.float32),
        "log_reviews": np.log1p(corpus["item_rating_reviews_count"]).astype(np.float32),
        "log_price": np.log1p(corpus["item_price"].clip(lower=0)).astype(np.float32),
        "title_len": corpus["item_title_raw"].str.len().astype(np.float32),
        "params_len": corpus["item_infm_params_text"].str.len().astype(np.float32),
        "desc_len": corpus["item_description_raw"].fillna("").str.len().astype(np.float32),
        "phone_hidden": corpus["item_is_phone_hidden"].astype(np.int8),
        "msg_forbidden": corpus["item_is_message_forbidden"].astype(np.int8),
    })


def variant_channels(fixes: bool, drop_channels=(), extra=()) -> list[str]:
    """Каналы варианта: базовые, исправления, каналы доп. dense-моделей, без отключённых."""
    chans = CHANNELS + (FIX_CHANNELS if fixes else [])
    chans += [extra_channel(ch, n) for n, _ in extra for ch in ["dense"] + (FIX_CHANNELS if fixes else [])]
    return [c for c in chans if c not in drop_channels]


def meta_kwargs(meta: dict) -> dict:
    """Аргументы build_features / кандидатов из метаданных варианта."""
    return {"fixes": meta.get("fixes", False), "model": meta.get("dense_model", MODEL),
            "name": meta.get("dense_name", "e5small"), "drop_channels": tuple(meta.get("drop_channels", ())),
            "extra": tuple(tuple(x) for x in meta.get("extra_dense", ()))}


def build_features(cands, queries, history, corpus, rrf_weights, *, fixes, truth=None, k=K_RANKER, seed=SEED,
                   model=MODEL, name="e5small", drop_channels=(), extra=()):
    """Строка на пару (запрос, объявление из пула). truth задаётся только для
    обучения: тогда негативы прореживаются (все из топ-NEG_TOP по RRF + NEG_RANDOM
    случайных), позитивы остаются все. fixes=False — вариант без исправлений."""
    channels = variant_channels(fixes, drop_channels, extra)
    cands = cands[cands["channel"].isin(channels)]
    queries = queries.assign(qtext=norm_text(queries["search_query"]))
    df = cands.loc[cands["rank"] <= k, ["qid", "item_id"]].drop_duplicates()
    # Ранги и скоры каналов по всей глубине кэша (до 500), не только внутри K.
    # По одному каналу за раз: pivot по всем каналам сразу не помещается в память.
    for ch in channels:
        d = cands.loc[cands["channel"] == ch, ["qid", "item_id", "rank", "score"]]
        d = d.astype({"rank": np.float32, "score": np.float32}).rename(columns={"rank": f"rank_{ch}", "score": f"score_{ch}"})
        df = df.merge(d, on=["qid", "item_id"], how="left")
    ranks = df[[f"rank_{ch}" for ch in channels]]
    df["n_channels"] = (ranks <= k).sum(axis=1)
    df["rrf"] = sum(rrf_weights.get(ch, 0) / (K_RRF + df[f"rank_{ch}"].fillna(10**6)) for ch in channels)
    df = df.sort_values(["qid", "rrf", "item_id"], ascending=[True, False, True], kind="stable")
    df["rrf_rank"] = df.groupby("qid").cumcount() + 1
    df["pool_size"] = df.groupby("qid")["item_id"].transform("size")

    if truth is not None:
        df["label"] = [i in truth[q] for q, i in zip(df["qid"], df["item_id"])]
        rng = np.random.default_rng(seed)
        r = rng.random(len(df))
        rest = ~df["label"] & (df["rrf_rank"] > NEG_TOP)
        # случайные NEG_RANDOM из «хвоста»: порог по доле хвоста в запросе
        tail_n = rest.groupby(df["qid"]).transform("sum").clip(lower=1)
        df = df[~rest | (r < NEG_RANDOM / tail_n)]

    # Объявление: признаки считаются один раз на корпус и присоединяются числами
    # (присоединять тексты к миллионам строк пула — это гигабайты памяти)
    df = df.merge(_item_features(corpus), on="item_id", how="left")

    if fixes:
        # Сходство с запросом для каждого кандидата, а не только для топа своего канала:
        # иначе у кандидатов из priors/pop_loc нет текстового сигнала вообще
        df = df.reset_index(drop=True)
        df["cos_dense"] = pair_cosine(queries, corpus, df[["qid", "item_id"]], model, name)
        df["bm25_title_all"] = bm25_pair_scores(queries, corpus, df[["qid", "item_id"]], "title")
        for n, m in extra:  # сходство и по доп. dense-модели
            df[f"cos_dense_{n}"] = pair_cosine(queries, corpus, df[["qid", "item_id"]], m, n)

    # Запрос и локация: центр локации запроса = медиана координат объявлений корпуса в ней
    q = queries.set_index("qid")
    df["q_loc"] = df["qid"].map(q["search_location_id"])
    df["q_words"] = df["qid"].map(q["qtext"].str.split().str.len())
    df["q_infm"] = df["qid"].map((norm_text(q["search_infm_params_text"]) != "").astype(int))
    df["loc_match"] = (df["item_location_id"] == df["q_loc"]).astype(int)
    cen = location_centroids(corpus)
    df["dist_km"] = haversine_km(df["q_loc"].map(cen["item_latitude"]), df["q_loc"].map(cen["item_longitude"]),
                                  df["item_latitude"], df["item_longitude"])

    # Популярность в истории как доля: у P2 и benchmark разный объём истории, счётчики были бы несравнимы
    pop = history["items"].explode().value_counts() / len(history)
    df["log_pop"] = np.log1p(1e5 * df["item_id"].map(pop).fillna(0))

    # Приоры: P(microcat | запрос), место microcat у запроса
    pm = microcat_probs(neighbors(queries, history), history)
    pm["mc_rank"] = pm.groupby("qid").cumcount() + 1
    df = df.merge(pm.rename(columns={"mc": "item_microcat_id", "p": "p_mc"}), on=["qid", "item_microcat_id"], how="left")
    df["p_mc"] = df["p_mc"].fillna(0)

    return df.sort_values(["qid", "item_id"]).reset_index(drop=True)


def feature_list(fixes: bool, drop_channels=(), drop_features=(), extra=()) -> list[str]:
    channels = variant_channels(fixes, drop_channels, extra)
    feats = ([f"{a}_{ch}" for ch in channels for a in ("rank", "score")] + BASE_FEATURES
             + (FIX_FEATURES + [f"cos_dense_{n}" for n, _ in extra] if fixes else []))
    return [f for f in feats if f not in drop_features]


RRF_FEATURES = ["rrf", "rrf_rank", "n_channels"]


BASE_FEATURES = (["n_channels", "rrf", "rrf_rank", "pool_size", "rating", "log_reviews", "log_price", "title_len",
               "params_len", "desc_len", "phone_hidden", "msg_forbidden", "q_words", "q_infm", "loc_match",
               "dist_km", "log_pop", "p_mc", "mc_rank"])
FEATURES = feature_list(False)  # набор признаков ранкера отправки №3


def fit(df, feats, loss, w_group, iterations=None, eval_df=None, cat_params=None):
    """YetiRank (CatBoostRanker) или Logloss (CatBoostClassifier); веса групп по локации."""
    from catboost import CatBoostClassifier, CatBoostRanker, Pool

    def pool(d):
        return Pool(d[feats].astype(np.float32), label=d["label"].astype(int), group_id=d["qid"],
                    group_weight=d["qid"].map(w_group).to_numpy() if loss == "YetiRank" else None,
                    weight=d["qid"].map(w_group).to_numpy() if loss == "Logloss" else None)

    params = dict(iterations=iterations or 2000, learning_rate=0.08, depth=6, random_seed=SEED,
                  thread_count=-1, verbose=200, allow_writing_files=False)
    params.update(cat_params or {})  # depth / learning_rate / l2_leaf_reg из перебора (выигрыша не дал)
    if eval_df is not None:
        # Ранняя остановка по собственной функции потерь на части для остановки
        # (Logloss для Logloss, PFound для YetiRank: скоры YetiRank без масштаба,
        # Logloss по ним бессмыслен). Recall@50 — только для выбора функции потерь,
        # на другой части P2: остановка по RecallAt срабатывала слишком рано.
        params.update(od_type="Iter", od_wait=100, use_best_model=True)
    m = (CatBoostRanker(loss_function="YetiRank", **params) if loss == "YetiRank"
         else CatBoostClassifier(loss_function="Logloss", **params))
    m.fit(pool(df), eval_set=pool(eval_df) if eval_df is not None else None)
    return m


def predict_top(m, df, feats, n=N_ANSWER):
    from catboost import CatBoostClassifier
    X = df[feats].astype(np.float32)
    # классификатору нужна вероятность (predict даёт метки), ранкеру — сырой скор
    s = m.predict_proba(X)[:, 1] if isinstance(m, CatBoostClassifier) else m.predict(X)
    d = df[["qid", "item_id"]].assign(s=s).sort_values(["qid", "s", "item_id"], ascending=[True, False, True], kind="stable")
    return d.groupby("qid").head(n).groupby("qid")["item_id"].agg(list).to_dict()


def part_data(sp, part):
    q = sp[sp["part"] == part]
    truth = {a: set(b) for a, b in zip(q["qid"], q["items"])}
    w = pd.Series(location_weights(q["search_location_id"]).to_numpy(), index=q["qid"])
    return q, truth, w


def p3_recall(m, meta, sp, corpus, hist, wts) -> pd.Series:
    """Recall по запросам P3 для обученной модели варианта meta."""
    q3, t3, _ = part_data(sp, "P3")
    kw = meta_kwargs(meta)
    c3 = cached_candidates("P3", q3, hist, corpus, kw["model"], kw["name"], kw["extra"])
    f3 = build_features(c3, q3, hist, corpus, wts, k=meta["k"], **kw)
    return recall_per_query(predict_top(m, f3, meta["features"]), t3).reindex(q3["qid"])


def train_variant(name: str, model: str, fixes: bool, drop_channels=(), drop_features=(),
                  suffix: str = "", eval_p3: bool = True, extra=(), p2_frac: float = 1.0) -> dict:
    """Обучение ранкера варианта: выбор функции потерь на P2, финал на всём P2, оценка на P3.

    drop_channels / drop_features — упрощённые варианты (без канала / группы признаков);
    eval_p3=False — только отбор на P2, P3 не трогаем.
    """
    tag = f"{name}{'_fix' if fixes else ''}{suffix}"
    sp = load_split()
    corpus = local_corpus(sp)
    hist = eval_history(sp)
    wts = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
    feats = feature_list(fixes, drop_channels, drop_features, extra)
    kw = dict(fixes=fixes, model=model, name=name, drop_channels=tuple(drop_channels), extra=tuple(extra))

    q2, t2, w2 = part_data(sp, "P2")
    c2 = cached_candidates("P2", q2, hist, corpus, model, name, extra)
    if p2_frac < 1.0:  # кривая обучения: та же процедура на доле запросов P2 (кандидаты из общего кэша)
        keep = np.random.default_rng(SEED + 1).permutation(np.sort(q2["qid"].to_numpy()))[:int(len(q2) * p2_frac)]
        q2 = q2[q2["qid"].isin(keep)]
        c2 = c2[c2["qid"].isin(keep)]
    # P2 по запросам: 60% обучение, 20% ранняя остановка (по функции потерь),
    # 20% выбор функции потерь по взвешенному Recall@50. Раздельно, чтобы выбор не шёл по той же выборке,
    # на которой остановились.
    perm = np.random.default_rng(SEED).permutation(np.sort(q2["qid"].to_numpy()))
    n5 = len(perm) // 5
    es_q, sel_q = set(perm[:n5]), set(perm[n5:2 * n5])
    full = build_features(c2, q2, hist, corpus, wts, truth=t2, **kw)
    full = full[full.groupby("qid")["label"].transform("any")]  # запросы без релевантного в пуле: учиться нечему
    tr = full[~full["qid"].isin(es_q | sel_q)]

    def unsampled(qs):
        d = build_features(c2[c2["qid"].isin(qs)], q2[q2["qid"].isin(qs)], hist, corpus, wts, **kw)
        d["label"] = [i in t2[q] for q, i in zip(d["qid"], d["item_id"])]
        return d
    es, sel = unsampled(es_q), unsampled(sel_q)
    sel_truth = {q: t2[q] for q in sel_q}
    del c2
    print(f"[{tag}] train {tr['qid'].nunique()} запросов, остановка {len(es_q)}, выбор потерь {len(sel_q)}")

    res = {}
    for loss in ("YetiRank", "Logloss"):
        m = fit(tr, feats, loss, w2, eval_df=es[es.groupby("qid")["label"].transform("any")])
        res[loss] = (recall_at_k(predict_top(m, sel, feats), sel_truth, weights=w2), m.get_best_iteration())
        print(f"[{tag}] P2 выбор потерь, {loss}: {res[loss][0]:.4f}, деревьев {res[loss][1] + 1}")
    loss = max(res, key=lambda k: res[k][0])
    print(f"[{tag}] выбран {loss}, деревьев {res[loss][1] + 1}")
    meta = {"loss": loss, "iterations": res[loss][1] + 1, "features": feats, "k": K_RANKER,
            "fixes": fixes, "dense_model": model, "dense_name": name,
            "drop_channels": list(drop_channels), "drop_features": list(drop_features), "p2_select": res[loss][0],
            "extra_dense": [list(x) for x in extra], "p2_frac": p2_frac}
    m = fit(full, feats, loss, w2, iterations=meta["iterations"])
    m.save_model(str(ARTIFACTS / f"ranker_{tag}.cbm"))
    (ARTIFACTS / f"ranker_meta_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if not eval_p3:
        return meta

    r = p3_recall(m, meta, sp, corpus, hist, wts)
    r.rename("recall").to_frame().to_parquet(ARTIFACTS / f"p3_recall_{tag}.parquet")
    _, _, w3 = part_data(sp, "P3")
    wv = w3.reindex(r.index).to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(r), (1000, len(r)))
    bs = (r.to_numpy()[idx] * wv[idx]).sum(1) / wv[idx].sum(1)
    print(f"[{tag}] P3 взв. {np.average(r, weights=wv):.4f} [{np.quantile(bs, .025):.4f}, {np.quantile(bs, .975):.4f}]")

    from catboost import Pool
    imp = pd.Series(m.get_feature_importance(Pool(full[feats].astype(np.float32), label=full["label"].astype(int),
                                                  group_id=full["qid"])), index=feats).sort_values(ascending=False)
    print(f"[{tag}] важность признаков, топ-12:\n{imp.head(12).round(2).to_string()}")
    return meta


if __name__ == "__main__":
    import sys
    # python -m src.ranker <имя кэша эмбеддингов> <модель> <fix|nofix>
    name, model, fx = (sys.argv[1:] + ["e5small", MODEL, "fix"])[:3] if len(sys.argv) > 1 else ("e5small", MODEL, "fix")
    train_variant(name, model, fx == "fix")
