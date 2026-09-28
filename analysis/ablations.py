"""Таблица абляций для README: ранкер без канала / без группы признаков -> Recall@50 на P3.

Признаки P2 (обучение) и P3 строятся один раз со всеми каналами. «Без канала»:
убираем его признаки и объявления, попавшие в пул только через него; RRF-признаки
(rrf, rrf_rank, n_channels, pool_size) пересчитываются без канала. Каждый вариант
переобучается на P2 той же функцией потерь (meta["loss"]) с тем же числом итераций (ranker_meta.json).
Негативы P2 прорежены по RRF со всеми каналами — небольшое упрощение.
Запуск: PYTHONPATH=. .venv/bin/python analysis/ablations.py [вариант]
вариант: sub3 (по умолчанию, ранкер отправки №3) или тег из artifacts/ranker_meta_<тег>.json.
"""
import sys
import json

import numpy as np
import pandas as pd

from src.candidates import K_RRF
from src.metrics import paired_bootstrap, recall_per_query
from src.candidates import cached_candidates
from src.ranker import CHANNELS, FIX_CHANNELS, build_features, fit, predict_top
from src.split import ARTIFACTS, eval_history, load_split, local_corpus, location_weights

sp = load_split(); corpus = local_corpus(sp); hist = eval_history(sp)
wts = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
TAG = sys.argv[1] if len(sys.argv) > 1 else "sub3"
meta = json.loads((ARTIFACTS / ("ranker_meta.json" if TAG == "sub3" else f"ranker_meta_{TAG}.json")).read_text())
K = meta["k"]
FIXES = meta.get("fixes", False)
KW = dict(fixes=FIXES, model=meta.get("dense_model", "intfloat/multilingual-e5-small"), name=meta.get("dense_name", "e5small"))
FEATURES = meta["features"]
CH = CHANNELS + (FIX_CHANNELS if FIXES else [])

q2 = sp[sp.part == "P2"]; t2 = {a: set(b) for a, b in zip(q2.qid, q2["items"])}
w2 = pd.Series(location_weights(q2.search_location_id).to_numpy(), index=q2.qid)
tr = build_features(cached_candidates("P2", q2, hist, corpus, KW["model"], KW["name"]), q2, hist, corpus, wts, truth=t2, **KW)
q3 = sp[sp.part == "P3"]; t3 = {a: set(b) for a, b in zip(q3.qid, q3["items"])}
te = build_features(cached_candidates("P3", q3, hist, corpus, KW["model"], KW["name"]), q3, hist, corpus, wts, **KW)
w3 = location_weights(q3.search_location_id).to_numpy()


def without_channel(df, ch):
    rest = [c for c in CH if c != ch]
    df = df[(df[[f"rank_{c}" for c in rest]] <= K).any(axis=1)].copy()
    w = {**wts, ch: 0.0}
    df["rrf"] = sum(w.get(c, 0) / (K_RRF + df[f"rank_{c}"].fillna(10**6)) for c in rest)
    df = df.sort_values(["qid", "rrf", "item_id"], ascending=[True, False, True], kind="stable")
    df["rrf_rank"] = df.groupby("qid").cumcount() + 1
    df["n_channels"] = (df[[f"rank_{c}" for c in rest]] <= K).sum(axis=1)
    df["pool_size"] = df.groupby("qid")["item_id"].transform("size")
    return df


def run(train, test, feats):
    train = train[train.groupby("qid")["label"].transform("any")]
    m = fit(train, feats, meta["loss"], w2, iterations=meta["iterations"], cat_params=meta.get("cat_params"))
    return recall_per_query(predict_top(m, test, feats), t3).reindex(q3.qid).to_numpy()


GROUPS = {
    "локация (loc_match, dist_km)": ["loc_match", "dist_km"],
    "популярность в истории (log_pop)": ["log_pop"],
    "P(подкатегория | запрос) (p_mc, mc_rank)": ["p_mc", "mc_rank"],
    "контент объявления": ["rating", "log_reviews", "log_price", "title_len", "params_len", "desc_len", "phone_hidden", "msg_forbidden"],
    "признаки запроса и пула": ["q_words", "q_infm", "pool_size"],
    "RRF-признаки (rrf, rrf_rank, n_channels)": ["rrf", "rrf_rank", "n_channels"],
}
if FIXES:
    GROUPS["сходство с запросом для всего пула (cos_dense, bm25_title_all)"] = ["cos_dense", "bm25_title_all"]
base = run(tr, te, FEATURES)
rows = [("ранкер, все каналы и признаки", np.average(base, weights=w3), 0.0, 0.0, 0.0)]
for ch in CH:
    feats = [f for f in FEATURES if f not in (f"rank_{ch}", f"score_{ch}")]
    r = run(without_channel(tr, ch), without_channel(te, ch), feats)
    rows.append((f"без канала {ch}", np.average(r, weights=w3), *paired_bootstrap(r, base, w3)))
    print(rows[-1], flush=True)
for name, g in GROUPS.items():
    r = run(tr, te, [f for f in FEATURES if f not in g])
    rows.append((f"без группы: {name}", np.average(r, weights=w3), *paired_bootstrap(r, base, w3)))
    print(rows[-1], flush=True)
tab = pd.DataFrame(rows, columns=["вариант", "Recall@50 P3 взв.", "разница", "95% от", "95% до"]).round(4)
tab.to_csv(ARTIFACTS / f"ablations_{TAG}.csv", index=False)
print(tab.to_string(index=False))
