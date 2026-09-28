"""Второй этап: cross-encoder поверх топ-150 первого ранкера.

1. first_stage(tag): оценки первого ранкера. P2 — out-of-fold (5 фолдов по запросам,
   ранкер с той же функцией потерь и числом деревьев), P3 и benchmark — ранкер на всём P2.
   Сохраняется топ-150 на запрос с сильнейшими признаками: artifacts/stage2/first_<часть>.parquet.
2. export_pairs(): пары топ-150 для Kaggle (query_id, item_id, текст запроса, текст объявления),
   в оба текста добавлен «город: X»; для P2 ещё label и n_rel (для Recall@50 в ноутбуке).
3. export_finetune_pairs(): n (по умолчанию 120k) случайных запросов истории P1 (с прореживанием): 1 позитив +
   4 dense-негатива (места 10–100 по пулу «история P1 + benchmark_items») + 3 позитива других
   запросов P1 той же подкатегории; известные позитивы того же текста исключены.
4. stage2_dataset / train_stage2: второй ранкер — оценка первого ранкера (OOF на P2), оценка
   cross-encoder, их ранги внутри топ-150 и несколько сильнейших старых признаков.
Запуск: python -m src.stage2 first [тег] | pairs | depth | finetune [n] | train2 [300]
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from src.candidates import build_candidates, cached_candidates
from src.data import item_dense_text, load, location_names, query_location_names, query_text
from src.metrics import recall_at_k
from src.split import ARTIFACTS, SEED, eval_history, load_split, local_corpus, location_weights

TOP = 150
STAGE2 = ARTIFACTS / "stage2"
KAGGLE = ARTIFACTS / "kaggle_rerank"
# Сильнейшие признаки первого ранкера по важности и абляциям (README): локация и dense
STRONG = ["dist_km", "loc_match", "score_dense_local", "rank_dense_local", "cos_dense", "bm25_title_all",
          "score_dense", "score_bm25_desc", "rrf_rank", "log_pop"]
# 120 000 запросов — выгрузка для последнего дообучения cross-encoder (python -m src.stage2 finetune 120000)
N_FINETUNE, N_NEG, NEG_FROM, NEG_TO = 120_000, 7, 10, 100


def _scores(m, meta, df):
    X = df[meta["features"]].astype(np.float32)
    return m.predict_proba(X)[:, 1] if meta["loss"] == "Logloss" else m.predict(X)


def _top(df, s, top=TOP):
    """Топ-top на запрос по оценке (tie-break item_id), с рангом first_rank."""
    d = df.assign(first_score=s).sort_values(["qid", "first_score", "item_id"], ascending=[True, False, True], kind="stable")
    d["first_rank"] = d.groupby("qid").cumcount() + 1
    d = d[d["first_rank"] <= top]
    return d[["qid", "item_id", "first_score", "first_rank"] + [c for c in STRONG if c in d.columns]].reset_index(drop=True)


def first_stage(tag: str, qids_limit: dict | None = None, top: int = TOP, suffix: str = "") -> dict:
    """qids_limit — {часть: число запросов} для быстрой проверки на маленьком наборе."""
    from make_answer import load_ranker
    from src.ranker import build_features, fit
    m, meta, kw = load_ranker(tag)
    sp = load_split(); corpus = local_corpus(sp); hist = eval_history(sp)
    wts = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
    lim = qids_limit or {}
    out = {}

    # P2: out-of-fold — у каждого запроса оценка ранкера, который его не видел
    q2 = sp[sp.part == "P2"].sort_values("qid").head(lim.get("P2", 10**9))
    t2 = {a: set(b) for a, b in zip(q2.qid, q2["items"])}
    w2 = pd.Series(location_weights(q2.search_location_id).to_numpy(), index=q2.qid)
    c2 = cached_candidates("P2", sp[sp.part == "P2"], hist, corpus, kw["model"], kw["name"], kw["extra"])
    c2 = c2[c2["qid"].isin(q2.qid)]
    full = build_features(c2, q2, hist, corpus, wts, truth=t2, **kw)
    full = full[full.groupby("qid")["label"].transform("any")]
    folds = np.array_split(np.random.default_rng(SEED).permutation(np.sort(q2.qid.to_numpy())), 5)
    parts = []
    for k, fq in enumerate(folds):
        fq = set(fq)
        mk = fit(full[~full["qid"].isin(fq)], meta["features"], meta["loss"], w2, iterations=meta["iterations"],
                 cat_params={"verbose": 0, **meta.get("cat_params", {})})
        fdf = build_features(c2[c2["qid"].isin(fq)], q2[q2.qid.isin(fq)], hist, corpus, wts, k=meta["k"], **kw)
        parts.append(_top(fdf, _scores(mk, meta, fdf), top))
        print(f"P2 фолд {k + 1}/5 готов", flush=True)
    out["P2"] = pd.concat(parts, ignore_index=True)

    # P3: ранкер на всём P2
    q3 = sp[sp.part == "P3"].sort_values("qid").head(lim.get("P3", 10**9))
    c3 = cached_candidates("P3", sp[sp.part == "P3"], hist, corpus, kw["model"], kw["name"], kw["extra"])
    f3 = build_features(c3[c3["qid"].isin(q3.qid)], q3, hist, corpus, wts, k=meta["k"], **kw)
    out["P3"] = _top(f3, _scores(m, meta, f3), top)

    # benchmark: история = весь train, корпус = benchmark_items (как в make_answer.py)
    bq = load("benchmark_queries").rename(columns={"query_id": "qid"}).head(lim.get("bench", 10**9))
    items = load("benchmark_items")
    cb = build_candidates(bq, sp, items, model=kw["model"], name=kw["name"], extra=kw["extra"])
    fb = build_features(cb, bq, sp, items, wts, k=meta["k"], **kw)
    out["bench"] = _top(fb, _scores(m, meta, fb), top)

    if not qids_limit:
        STAGE2.mkdir(exist_ok=True)
        for part, d in out.items():
            d.to_parquet(STAGE2 / f"first_{part}{suffix}.parquet", index=False)
        (STAGE2 / f"first_meta{suffix}.json").write_text(json.dumps({"tag": tag, "top": top}, ensure_ascii=False))
    return out


def _names(history):
    cols = ["item_id", "item_location_id", "item_infm_params_text"]
    it = pd.concat([load("benchmark_items", cols), load("train", cols)]).drop_duplicates("item_id")
    names = location_names(it)
    item_loc = it.set_index("item_id")["item_location_id"]
    return names, query_location_names(names, history, item_loc)


def _q_text(q, qnames):
    return query_text(q) + " | город: " + q["search_location_id"].map(qnames).fillna("неизвестно")


def _i_text(items, inames):
    return item_dense_text(items) + " | город: " + items["item_location_id"].map(inames).fillna("неизвестно")


def export_pairs(firsts: dict | None = None, out_dir=KAGGLE) -> None:
    """firsts — {часть: топ-150} (по умолчанию из artifacts/stage2/), out_dir — куда писать."""
    sp = load_split(); hist = eval_history(sp)
    out_dir.mkdir(exist_ok=True)
    for part in ("P2", "P3", "bench"):
        first = firsts[part] if firsts else pd.read_parquet(STAGE2 / f"first_{part}.parquet")
        if part == "bench":
            q = load("benchmark_queries").rename(columns={"query_id": "qid"})
            items, history = load("benchmark_items"), sp  # как в make_answer.py
        else:
            q, items, history = sp[sp.part == part], local_corpus(sp), hist
        inames, qnames = _names(history)
        qt = pd.Series(_q_text(q, qnames).to_numpy(), index=q["qid"])
        it = pd.Series(_i_text(items, inames).to_numpy(), index=items["item_id"])
        d = pd.DataFrame({"query_id": first["qid"], "item_id": first["item_id"], "query_text": first["qid"].map(qt),
                          "item_text": first["item_id"].map(it), "first_rank": first["first_rank"]})
        assert d["query_text"].notna().all() and d["item_text"].notna().all()
        if part == "P2":  # разметка только для P2: ноутбук сравнивает модели по Recall@50 на P2
            truth = dict(zip(q["qid"], map(set, q["items"])))
            d["label"] = [int(i in truth[qq]) for qq, i in zip(d["query_id"], d["item_id"])]
            d["n_rel"] = d["query_id"].map(lambda qq: len(truth[qq]))
            w = pd.Series(location_weights(q["search_location_id"]).to_numpy(), index=q["qid"])
            d["w"] = d["query_id"].map(w)  # вес запроса по локации: Recall@50 в ноутбуке как локально
        path = out_dir / f"pairs_{part}.parquet"
        d.to_parquet(path, index=False, compression="zstd")
        print(f"{path.name}: {len(d)} пар, {d['query_id'].nunique()} запросов, {path.stat().st_size / 1e6:.0f} МБ")


def export_finetune_pairs(model="artifacts/e5-base-ft", name="e5baseft", n=N_FINETUNE, out_dir=KAGGLE) -> None:
    """Пары для дообучения cross-encoder только из истории P1 (с прореживанием).

    На каждый позитив 7 негативов из ТОГО ЖЕ пула, что и позитивы, чтобы модель не выучила
    «стиль объявления из истории = релевантно» (позитивы из истории, негативы только из
    benchmark_items различались по тексту с AUC 0.57):
    - 4 dense-негатива с мест 10–100 по пулу «история P1 + benchmark_items»;
    - 3 случайных позитива других запросов P1 из той же подкатегории, что и позитив.
    Известные позитивы того же текста запроса в истории исключаются. Часть объявлений,
    вставленных в локальный корпус как релевантные P2/P3 (3 083 из 12 541), в пуле есть: их
    выбирали и в истории P1. Это законная история P1 (как и для benchmark после прореживания),
    разметка P2/P3 здесь не используется.
    """
    from src.channels.dense import _encode, corpus_embeddings, prefixes
    sp = load_split(); hist = eval_history(sp)
    rng = np.random.default_rng(SEED)
    q = hist.iloc[np.sort(rng.choice(len(hist), n, replace=False))].reset_index(drop=True)
    known = hist.groupby("qtext")["items"].agg(lambda s: set().union(*map(set, s)))  # все позитивы текста в истории
    pos = [rng.choice(sorted(its)) for its in q["items"]]

    # Пул объявлений: история P1 + benchmark_items, у всех dense-эмбеддинги e5-base-ft
    bench = load("benchmark_items")
    cols = list(bench.columns)
    h_ids = sorted(set(hist["items"].explode()) - set(bench["item_id"]))
    h_items = load("train", cols, filters=[("item_id", "in", h_ids)]).drop_duplicates("item_id").sort_values("item_id")
    ids_b, emb_b = corpus_embeddings(bench, name, model)
    ids_h, emb_h = corpus_embeddings(h_items, f"{name}_hist", model)
    pool_ids, pool_emb = np.r_[ids_b, ids_h], np.vstack([emb_b, emb_h])
    del emb_b, emb_h
    qe = _encode((prefixes(model)[0] + query_text(q)).tolist(), model)
    dense_neg = []
    for s0 in range(0, len(q), 500):  # куски: 500 × 500k float32 = 1 ГБ
        sim = qe[s0:s0 + 500] @ pool_emb.T
        top = np.argpartition(-sim, NEG_TO, axis=1)[:, :NEG_TO]
        for j, row in enumerate(top):
            row = row[np.argsort(-sim[j, row], kind="stable")]
            kn = known.get(q["qtext"].iat[s0 + j], set())
            cand = [pool_ids[i] for i in row[NEG_FROM - 1:] if pool_ids[i] not in kn]  # места 10–100
            dense_neg.append(list(rng.choice(cand, min(4, len(cand)), replace=False)))

    # Позитивы других запросов P1 той же подкатегории
    mc = pd.concat([bench[["item_id", "item_microcat_id"]], h_items[["item_id", "item_microcat_id"]]]).drop_duplicates("item_id")
    mc = mc.set_index("item_id")["item_microcat_id"]
    hist_items = np.array(sorted(set(hist["items"].explode())))
    mcv = mc.reindex(hist_items).to_numpy()
    by_mc = {k: hist_items[ix] for k, ix in pd.Series(mcv).groupby(mcv).indices.items()}
    mc_neg = []
    for j, p in enumerate(pos):
        kn = known.get(q["qtext"].iat[j], set())
        cand = by_mc.get(mc.get(p), np.array([]))
        pick = [c for c in rng.permutation(cand)[:50] if c not in kn][:3]
        mc_neg.append(pick)

    inames, qnames = _names(hist)
    all_items = pd.concat([bench, h_items]).drop_duplicates("item_id")
    it_text = pd.Series(_i_text(all_items, inames).to_numpy(), index=all_items["item_id"])
    qt = _q_text(q, qnames).to_numpy()
    rows = [(qt[j], pos[j], 1, "позитив") for j in range(len(q))]
    rows += [(qt[j], x, 0, "dense 10–100") for j in range(len(q)) for x in dense_neg[j]]
    rows += [(qt[j], x, 0, "подкатегория") for j in range(len(q)) for x in mc_neg[j]]
    d = pd.DataFrame(rows, columns=["query_text", "item_id", "label", "kind"])
    d["item_text"] = d["item_id"].map(it_text)
    assert d["item_text"].notna().all()
    d = d.sample(frac=1, random_state=SEED)[["query_text", "item_text", "label", "item_id", "kind"]].reset_index(drop=True)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "finetune_pairs.parquet"
    d.to_parquet(path, index=False, compression="zstd")
    print(f"{path.name}: {len(d)} пар ({d['label'].sum()} позитивов), {d['kind'].value_counts().to_dict()}, {path.stat().st_size / 1e6:.0f} МБ")


# ---------- второй ранкер ----------

STAGE2_BASE = ["first_score", "first_rank"] + STRONG


def stage2_dataset(first: pd.DataFrame, ce: dict) -> pd.DataFrame:
    """first — топ-150 первого ранкера (OOF для P2); ce — {"zeroshot"|"finetuned": DataFrame(qid, item_id, ce_score)},
    каждая модель — если её оценки есть. На модель три признака: оценка, разница с максимумом по запросу
    (насколько объявление отстаёт от лучшего кандидата этого запроса) и ранг внутри запроса."""
    d = first.copy()
    for name, sc in ce.items():
        col = f"ce_{name}"
        d = d.merge(sc[["qid", "item_id", "ce_score"]].rename(columns={"ce_score": col}), on=["qid", "item_id"], how="left")
        assert d[col].notna().all(), f"нет оценки {name} для части пар"
        d[f"{col}_gap"] = d[col] - d.groupby("qid")[col].transform("max")
        d[f"{col}_rank"] = d.groupby("qid")[col].rank(ascending=False, method="first")
    return d.sort_values(["qid", "first_rank"]).reset_index(drop=True)


def stage2_features(d: pd.DataFrame) -> list[str]:
    return [f for f in STAGE2_BASE if f in d.columns] + [c for c in d.columns if c.startswith(("ce_", "flt_"))]


ENS_MEMBERS = [("Logloss", 42), ("Logloss", 43), ("Logloss", 44), ("YetiRank", 42), ("YetiRank", 43), ("YetiRank", 44)]


def _mean_rank(d: pd.DataFrame, scores: list) -> np.ndarray:
    """Средний ранг моделей внутри запроса со знаком минус (больше — лучше)."""
    return -np.mean([d[["qid"]].assign(s=s).groupby("qid")["s"].rank(ascending=False, method="first").to_numpy()
                     for s in scores], axis=0)


def _score(m, d, feats):
    X = d[feats].astype(np.float32)
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)


def _top50(d, s):
    x = d[["qid", "item_id"]].assign(s=s).sort_values(["qid", "s", "item_id"], ascending=[True, False, True], kind="stable")
    return x.groupby("qid").head(50).groupby("qid")["item_id"].agg(list).to_dict()


def train_stage2(d2: pd.DataFrame, truth2: dict, w2: pd.Series, d3: pd.DataFrame, truth3: dict, w3: pd.Series, drop=(),
                 ensemble: bool = False, p2_frac: float = 1.0):
    """Та же процедура, что у первого ранкера: 60% обучение / 20% остановка / 20% выбор потерь по
    взвешенному Recall@50, финал на всём P2. ensemble — среднее рангов 6 моделей (Logloss и YetiRank,
    3 seed), каждая со своей ранней остановкой. p2_frac — доля запросов P2 (кривая обучения).
    Возвращает модель (или список моделей), сводку и recall P3 по запросам."""
    from src.metrics import recall_per_query
    from src.ranker import fit
    feats = [f for f in stage2_features(d2) if f not in drop]
    if p2_frac < 1.0:
        keep = np.random.default_rng(SEED + 1).permutation(np.sort(d2["qid"].unique()))[:int(d2["qid"].nunique() * p2_frac)]
        d2 = d2[d2["qid"].isin(set(keep))]
    d2 = d2.assign(label=[i in truth2[q] for q, i in zip(d2["qid"], d2["item_id"])])
    perm = np.random.default_rng(SEED).permutation(np.sort(d2["qid"].unique()))
    n5 = len(perm) // 5
    es_q, sel_q = set(perm[:n5]), set(perm[n5:2 * n5])
    pos = d2[d2.groupby("qid")["label"].transform("any")]
    tr, es = pos[~pos["qid"].isin(es_q | sel_q)], pos[pos["qid"].isin(es_q)]
    sel = d2[d2["qid"].isin(sel_q)]
    sel_truth = {q: truth2[q] for q in sel_q}
    if ensemble:
        found, ss = [], []
        for loss, seed in ENS_MEMBERS:
            m = fit(tr, feats, loss, w2, eval_df=es, cat_params={"verbose": 0, "random_seed": seed})
            found.append((loss, seed, m.get_best_iteration() + 1)); ss.append(_score(m, sel, feats))
        p2_sel = recall_at_k(_top50(sel, _mean_rank(sel, ss)), sel_truth, weights=w2)
        models = [fit(pos, feats, loss, w2, iterations=it, cat_params={"verbose": 0, "random_seed": seed}) for loss, seed, it in found]
        pred3 = _top50(d3, _mean_rank(d3, [_score(m, d3, feats) for m in models]))
        info = {"loss": "ensemble", "members": found, "features": feats, "p2_select": p2_sel}
        m = models
    else:
        res = {}
        for loss in ("YetiRank", "Logloss"):
            mm = fit(tr, feats, loss, w2, eval_df=es, cat_params={"verbose": 0})
            res[loss] = (recall_at_k(_top50(sel, _score(mm, sel, feats)), sel_truth, weights=w2), mm.get_best_iteration() + 1)
        loss = max(res, key=lambda k: res[k][0])
        m = fit(pos, feats, loss, w2, iterations=res[loss][1], cat_params={"verbose": 0})
        pred3 = _top50(d3, _score(m, d3, feats))
        info = {"loss": loss, "iterations": res[loss][1], "features": feats, "p2_select": res[loss][0]}
    r3 = recall_per_query(pred3, truth3)
    info["p3"] = float((r3 * w3.reindex(r3.index)).sum() / w3.reindex(r3.index).sum())
    return m, info, r3


# ---------- дополнительные признаки второго ранкера ----------

FILTER_KEYS = {"Вид услуги": "vid", "Тип услуги": "tip", "Кто оказывает услуги": "kto"}
# Ключи-разделители в search_infm_params_text: значение ключа идёт до следующего ключа.
# «Тип услуги автосервиса» — отдельный ключ, «Тип услуги» его не захватывает.
_DELIMS = ["Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Кто оказывает услуги", "Срочная услуга (мультистатус)",
           "Онлайн-запись", "Где вы оказываете услуги", "Чем вы занимаетесь", "Поиск по слотам", "Предмет или специальность"]
_DELIM_RE = "|".join(re.escape(k) + (r"(?! автосервиса)" if k == "Тип услуги" else "") for k in _DELIMS)


def parse_filters(text: str) -> dict:
    """{ключ: [значения]} для FILTER_KEYS из текста фильтров запроса."""
    out = {}
    if not isinstance(text, str) or not text:
        return out
    for m in re.finditer(rf"(?:^| )({_DELIM_RE}) (.+?)(?= (?:{_DELIM_RE})(?: |$)|$)", text):
        if m.group(1) in FILTER_KEYS:
            out.setdefault(m.group(1), []).append(m.group(2).strip())
    return out


def add_filter_features(d: pd.DataFrame, q_params: pd.Series, i_params: pd.Series) -> pd.DataFrame:
    """Совпадение фильтров запроса с параметрами объявления (смысл запроса часто в фильтрах): на ключ 1 — в параметрах
    объявления есть «ключ значение» для какого-либо значения ключа из запроса, 0 — нет, NaN — ключа в
    запросе нет; плюс число совпавших и несовпавших ключей."""
    parsed = {q: parse_filters(t) for q, t in q_params.items()}
    it = d["item_id"].map(i_params).fillna("").to_numpy()
    qs = d["qid"].to_numpy()
    d = d.copy()
    for key, slug in FILTER_KEYS.items():
        col = np.full(len(d), np.nan, dtype=np.float32)
        for j, (q, text) in enumerate(zip(qs, it)):
            vals = parsed.get(q, {}).get(key)
            if vals:
                col[j] = float(any(f"{key} {v}" in text for v in vals))
        d[f"flt_{slug}"] = col
    f = d[[f"flt_{s}" for s in FILTER_KEYS.values()]]
    d["flt_match"], d["flt_miss"] = (f == 1).sum(axis=1).astype(np.float32), (f == 0).sum(axis=1).astype(np.float32)
    return d


def add_ce_extra(d: pd.DataFrame, groups=("inter", "norm"), thr: float | None = None, name: str = "zeroshot") -> pd.DataFrame:
    """inter — оценка cross-encoder × совпадение локации и × расстояние; norm — разность со средним
    топ-5 оценок запроса и доля кандидатов запроса с оценкой выше порога thr (порог выбирается на P2)."""
    d = d.copy()
    ce = d[f"ce_{name}"]
    if "inter" in groups:
        d["ce_x_loc"] = ce * d["loc_match"]
        d["ce_x_dist"] = ce * d["dist_km"]
    if "norm" in groups:
        top5 = d.groupby("qid")[f"ce_{name}"].transform(lambda s: s.nlargest(5).mean())
        d["ce_minus_top5"] = ce - top5
        if thr is not None:
            d["ce_share_thr"] = (ce > thr).groupby(d["qid"]).transform("mean").astype(np.float32)
    return d


# Оценки cross-encoder: выход notebooks/kaggle_3_score_reranker.ipynb (места 1–150 и 151–300)
CE_DIRS = {"zeroshot": ARTIFACTS / "kaggle_output_B_zeroshot"}
STAGE2_TAG = "zeroshot"


def load_ce(part: str, first: pd.DataFrame, dirs: dict) -> dict:
    """Оценки cross-encoder для части и проверка, что они посчитаны именно для этих пар топа.

    Пары (запрос, объявление) в файлах оценок должны совпадать с текущим топом первого этапа
    один в один и без повторов; иначе оценки от другой выгрузки пар и второй ранкер получил бы
    чужие признаки. Значение dirs — папка или список папок (глубина: места 1–150 и 151–300).
    """
    out = {}
    for name, d in dirs.items():
        if isinstance(d, (list, tuple)):   # глубина: оценки мест 1–150 и 151–300 из разных выгрузок
            parts = [load_ce_raw(part, x) for x in d]
            sc = pd.concat(parts, ignore_index=True)
            path = " + ".join(f"{x}/scores_{part}.parquet" for x in d)
        else:
            sc, path = load_ce_raw(part, d), f"{d}/scores_{part}.parquet"
        want = set(zip(first["qid"], first["item_id"]))
        got = set(zip(sc["qid"], sc["item_id"]))
        if want != got or sc["ce_score"].isna().any() or len(sc) != len(got):
            raise ValueError(f"оценки {path} не соответствуют текущему топу {part}: пар в выгрузке {len(want)}, "
                             f"в оценках {len(got)}, общих {len(want & got)}. Пересоберите пары и оценки (README, «Второй этап»)")
        out[name] = sc
    return out


def load_ce_raw(part: str, d) -> pd.DataFrame:
    path = f"{d}/scores_{part}.parquet"
    try:
        return pd.read_parquet(path).rename(columns={"query_id": "qid"})
    except FileNotFoundError:
        raise FileNotFoundError(
            f"нет оценок cross-encoder {path}. Шаги (README, «Второй этап»): python -m src.stage2 first; "
            f"python -m src.stage2 pairs; загрузить artifacts/kaggle_rerank/pairs_*.parquet на Kaggle и выполнить "
            f"notebooks/kaggle_3_score_reranker.ipynb; scores_*.parquet из Output положить в {d}/") from None


# Глубина топа первого этапа: 150 — оценки одной выгрузки; 300 — места 1–150 и 151–300 из двух
DEPTHS = {150: ("", CE_DIRS),
          300: ("_300", {"zeroshot": [ARTIFACTS / "kaggle_output_B_zeroshot", ARTIFACTS / "kaggle_output_B_depth"]})}


def train2(dirs: dict = CE_DIRS, tag: str = STAGE2_TAG, drop=(), save: bool = True, depth: int = 150):
    """Второй ранкер: обучение на P2 (OOF первого этапа), проверка на P3. Сохраняется для make_answer.py.
    depth — глубина топа первого этапа (150 или 300; для 300 dirs берутся из DEPTHS)."""
    sp = load_split()
    suffix = DEPTHS[depth][0]
    if depth != 150:
        dirs = DEPTHS[depth][1]
    first = {p: pd.read_parquet(STAGE2 / f"first_{p}{suffix}.parquet") for p in ("P2", "P3")}
    d = {p: stage2_dataset(first[p], load_ce(p, first[p], dirs)) for p in ("P2", "P3")}
    truth = {p: dict(zip(sp.loc[sp.part == p, "qid"], map(set, sp.loc[sp.part == p, "items"]))) for p in ("P2", "P3")}
    w = {p: pd.Series(location_weights(sp.loc[sp.part == p, "search_location_id"]).to_numpy(), index=sp.loc[sp.part == p, "qid"])
         for p in ("P2", "P3")}
    m, info, r3 = train_stage2(d["P2"], truth["P2"], w["P2"], d["P3"], truth["P3"], w["P3"], drop=drop)
    if save:
        m.save_model(str(STAGE2 / f"ranker2_{tag}.cbm"))
        rel = lambda v: [str(x.relative_to(ARTIFACTS.parent)) for x in v] if isinstance(v, (list, tuple)) else str(v.relative_to(ARTIFACTS.parent))
        meta = {**info, "ce_dirs": {k: rel(v) for k, v in dirs.items()}, "top": depth,
                "first": json.loads((STAGE2 / f"first_meta{suffix}.json").read_text())}
        (STAGE2 / f"ranker2_{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        r3.rename("recall").to_frame().to_parquet(ARTIFACTS / f"p3_recall_stage2_{tag}.parquet")
    return m, info, r3


if __name__ == "__main__":
    # python -m src.stage2 first [тег первого ранкера] | pairs | depth | finetune [n запросов] | train2 [глубина: 300]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "first"

    def depth():
        """Топ-300 первого этапа (first_*_300.parquet) и пары мест 151–300 -> artifacts/kaggle_rerank_depth/:
        их оценивает notebooks/kaggle_3_score_reranker.ipynb, выход кладётся в artifacts/kaggle_output_B_depth/."""
        firsts = first_stage("e5baseft_fix_bge_nopriors", top=300, suffix="_300")
        export_pairs({p: d[d["first_rank"] > TOP] for p, d in firsts.items()}, ARTIFACTS / "kaggle_rerank_depth")
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    {"first": lambda: first_stage(arg or "e5baseft_fix_bge_nopriors"),
     "pairs": export_pairs,
     "depth": depth,
     "finetune": lambda: export_finetune_pairs(n=int(arg) if arg else N_FINETUNE),
     "train2": lambda: print({k: v for k, v in (train2(depth=int(arg), tag=f"zeroshot_d{arg}") if arg else train2())[1].items()
                              if k != "features"})}[cmd]()
