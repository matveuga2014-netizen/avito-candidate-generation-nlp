"""Каналы на P2/P3: Recall@50 каждого, recall пула, подбор весов RRF на P2, проверка на P3.

Запуск: python -m src.eval_channels
Кандидаты кэшируются в artifacts/cands_<part>.parquet, веса RRF в artifacts/rrf_weights.json.
Главная цифра — Recall@50, взвешенный по локации (калибровка по отправке №1, см. README).
"""
import json

import numpy as np
import pandas as pd

from src.candidates import build_candidates, rrf
from src.data import load, norm_text
from src.metrics import recall_at_k, recall_per_query
from src.split import ARTIFACTS, eval_history, load_split, local_corpus, location_weights

GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
CELL_BINS = [-2, -0.5, 0, 9, 49, 199, 10**9]


def cell_size(queries, history, corpus) -> pd.Series:
    """Диагностика: размер ячейки (топ-1 предсказанная microcat, локация запроса) в корпусе.

    В P3 ячейки втрое меньше, чем в benchmark (медиана 8 против 27), и каналы,
    привязанные к локации, там переоцениваются. Считается без разметки.
    """
    from src.channels.logs import neighbors
    mc = load("train", ["item_id", "item_microcat_id"]).drop_duplicates("item_id").set_index("item_id")["item_microcat_id"]
    ex = history[["qtext", "items"]].explode("items")
    ex["mc"] = ex["items"].map(mc)
    exm = ex.groupby(["qtext", "mc"]).size().rename("cnt").reset_index()
    pm = neighbors(queries, history).merge(exm, left_on="htext", right_on="qtext").assign(w=lambda d: d["sim"] * d["cnt"])
    pm = pm.groupby(["qid", "mc"], as_index=False)["w"].sum()
    top = pm.sort_values(["qid", "w", "mc"], ascending=[True, False, True]).drop_duplicates("qid")
    top = top.merge(queries[["qid", "search_location_id"]], on="qid")
    cell = corpus.groupby(["item_microcat_id", "item_location_id"]).size()
    top["cell"] = [cell.get((m, l), 0) for m, l in zip(top["mc"], top["search_location_id"])]
    return top.set_index("qid")["cell"].reindex(queries["qid"]).fillna(-1)


def get_cands(part: str, sp, corpus) -> pd.DataFrame:
    """Каналы дня 1 (без dense_local): веса RRF и цифры журнала посчитаны на них."""
    from src.candidates import cached_candidates
    from src.ranker import CHANNELS
    c = cached_candidates(part, sp[sp["part"] == part], eval_history(sp), corpus)
    return c[c["channel"].isin(CHANNELS)]


def truth_weights(sp, part):
    q = sp[sp["part"] == part]
    truth = {a: set(b) for a, b in zip(q["qid"], q["items"])}
    w = pd.Series(location_weights(q["search_location_id"]).to_numpy(), index=q["qid"])
    return q, truth, w


def pool_recall(cands, truth, w, K):
    pool = cands[cands["rank"] <= K].groupby("qid")["item_id"].agg(lambda s: list(dict.fromkeys(s))).to_dict()
    return recall_at_k(pool, truth, k=10**9, weights=w)


def tune(cands, truth, w, channels, max_rank=200):
    """Покоординатный подбор весов RRF по сетке, 3 прохода; старт — все веса 1.

    Вклады 1/(K_RRF + rank) считаются один раз в матрицу (пара запрос–объявление
    × канал), шаг сетки = одно умножение и сортировка. Ранги обрезаны до
    max_rank и фолбэк не добивается: это только для подбора, итоговые цифры
    считает настоящий rrf().
    """
    from src.candidates import K_RRF, N_ANSWER
    c = cands[(cands["rank"] <= max_rank) & cands["channel"].isin(channels)]
    piv = c.pivot_table(index=["qid", "item_id"], columns="channel", values="rank", aggfunc="min")
    M = np.nan_to_num(1.0 / (K_RRF + piv[channels].to_numpy(dtype=np.float32)), nan=0.0)
    qids = piv.index.get_level_values(0).to_numpy()
    items = piv.index.get_level_values(1).to_numpy()
    qcode = pd.factorize(qids, sort=True)[0]
    icode = pd.factorize(items, sort=True)[0]  # tie-break по item_id, как в rrf()
    rel = np.fromiter((i in truth.get(q, ()) for q, i in zip(qids, items)), bool, len(qids))
    uq = np.sort(np.unique(qids))
    nrel = np.array([len(truth[q]) for q in uq], dtype=float)
    wq = w.reindex(uq).to_numpy()
    # запросы без кандидатов тоже в знаменателе (их recall = 0)
    wsum = w.sum()

    def score(wts):
        s_ = M @ np.array([wts[ch] for ch in channels], dtype=np.float32)
        o = np.lexsort((icode, -s_, qcode))
        qs = qcode[o]
        start = np.searchsorted(qs, qs, side="left")
        top = (np.arange(len(o)) - start) < N_ANSWER
        hits = np.bincount(qs[top], weights=rel[o][top], minlength=len(uq))
        return float((hits / nrel * wq).sum() / wsum)

    wts = {ch: 1.0 for ch in channels}
    best = score(wts)
    for _ in range(3):
        for ch in channels:
            for g in GRID:
                trial = {**wts, ch: g}
                if not any(trial.values()):
                    continue
                r = score(trial)
                if r > best + 1e-9:
                    best, wts = r, trial
    return wts, best


def main():
    sp = load_split()
    corpus = local_corpus(sp)
    res = {}
    for part in ("P2", "P3"):
        cands = get_cands(part, sp, corpus)
        q, truth, w = truth_weights(sp, part)
        rows = []
        for ch, d in cands.groupby("channel"):
            pred = d.sort_values(["qid", "rank"]).groupby("qid")["item_id"].agg(list).to_dict()
            rows.append({"channel": ch, "R@50 взв.": recall_at_k(pred, truth, weights=w), "R@50": recall_at_k(pred, truth),
                         "R@500 взв.": recall_at_k(pred, truth, 500, w)})
        print(f"\n== {part}: каналы\n", pd.DataFrame(rows).round(4).to_string(index=False))
        print("пул (объединение топ-K каналов), взв.:", {K: round(pool_recall(cands, truth, w, K), 4) for K in (50, 100, 300, 500)})
        res[part] = (cands, truth, w, q)

    channels = sorted(res["P2"][0]["channel"].unique())
    wts, best = tune(res["P2"][0], res["P2"][1], res["P2"][2], channels)
    print("\nвеса RRF (подобраны на P2):", wts, "| P2 взв.", round(best, 4))
    (ARTIFACTS / "rrf_weights.json").write_text(json.dumps(wts, ensure_ascii=False, indent=1))

    cands, truth, w, q = res["P3"]
    pred = rrf(cands, wts)
    print("P3 RRF равные веса: взв.", round(recall_at_k(rrf(cands, {c: 1.0 for c in channels}), truth, weights=w), 4))
    print("P3 RRF подобранные веса: взв.", round(recall_at_k(pred, truth, weights=w), 4), "| без весов", round(recall_at_k(pred, truth), 4))
    r = recall_per_query(pred, truth).reindex(q["qid"]).to_numpy()
    empty = (norm_text(q["search_infm_params_text"]) == "").to_numpy()
    wv = w.reindex(q["qid"]).to_numpy()
    for name, m in (("пустой infm_params", empty), ("непустой infm_params", ~empty)):
        print(f"  срез {name}: n={m.sum()}, взв. {np.average(r[m], weights=wv[m]):.4f}")
    # Бутстрэп (взвешенный) для интервала итоговой цифры
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(r), (1000, len(r)))
    bs = (r[idx] * wv[idx]).sum(1) / wv[idx].sum(1)
    print(f"  95% интервал (бутстрэп 1000): [{np.quantile(bs, .025):.4f}, {np.quantile(bs, .975):.4f}]")

    # Диагностика: веса по локации × поправка на размер ячейки под benchmark
    bq = load("benchmark_queries").rename(columns={"query_id": "qid"})
    bq["qtext"] = norm_text(bq["search_query"])
    b_bucket = pd.cut(cell_size(bq, sp, load("benchmark_items")), CELL_BINS).value_counts(normalize=True)
    p_bucket = pd.cut(cell_size(q, eval_history(sp), corpus), CELL_BINS)
    p_share = pd.Series(wv, index=p_bucket.index).groupby(p_bucket, observed=False).sum() / wv.sum()
    wc = wv * p_bucket.map(b_bucket / p_share).astype(float).fillna(0).to_numpy()
    print("\nдиагностика, P3 с весами локация × размер ячейки:")
    print("  RRF:", round(np.average(r, weights=wc), 4))
    for ch, d in cands.groupby("channel"):
        rc = recall_per_query(d.sort_values(["qid", "rank"]).groupby("qid")["item_id"].agg(list).to_dict(), truth).reindex(q["qid"]).to_numpy()
        print(f"  {ch}: {np.average(rc, weights=wc):.4f}")


if __name__ == "__main__":
    main()
