"""Анализ промахов ранкера на P3 (день 3, пункт 1).

Промах = релевантное объявление запроса P3 не попало в топ-50 ранкера.
1) Автоматические признаки по всем промахам (доли).
2) 50 случайных промахов (seed 42) с деталями -> artifacts/errors_p3.txt для ручной разметки.
Запуск: PYTHONPATH=. .venv/bin/python analysis/error_analysis.py
"""
import json

import numpy as np
import pandas as pd

from make_answer import load_ranker
from src.channels.logs import microcat_probs, neighbors
from src.data import haversine_km, norm_text
from src.candidates import cached_candidates
from src.ranker import CHANNELS, build_features
from src.split import ARTIFACTS, SEED, eval_history, load_split, local_corpus

sp = load_split(); hist = eval_history(sp); corpus = local_corpus(sp)
p3 = sp[sp["part"] == "P3"].copy()
m, meta, _ = load_ranker("sub3")  # цифры README посчитаны для ранкера отправки №3
wts = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
cands = cached_candidates("P3", p3, hist, corpus)
cands = cands[cands["channel"].isin(CHANNELS)]
f = build_features(cands, p3, hist, corpus, wts, k=meta["k"], fixes=False)
f["s"] = m.predict_proba(f[meta["features"]].astype(np.float32))[:, 1]
f = f.sort_values(["qid", "s", "item_id"], ascending=[True, False, True])
f["pos"] = f.groupby("qid").cumcount() + 1

# Пары (запрос, релевантное) и где релевантное оказалось
rel = p3[["qid", "qtext", "search_query", "search_infm_params_text", "search_location_id", "items"]].explode("items").rename(columns={"items": "item_id"})
rel = rel.merge(f[["qid", "item_id", "pos", "dist_km"] + [f"rank_{c}" for c in CHANNELS]], on=["qid", "item_id"], how="left")
rel["miss"] = ~(rel["pos"] <= 50)
it = corpus.set_index("item_id")
rel["rel_title"] = rel["item_id"].map(it["item_title_raw"])
rel["rel_loc"] = rel["item_id"].map(it["item_location_id"])
rel["rel_mc"] = rel["item_id"].map(it["item_microcat_id"])

pm = microcat_probs(neighbors(p3, hist), hist)
top3 = pm.groupby("qid").head(3).groupby("qid")["mc"].agg(set)
hist_texts = set(hist["qtext"])
rel["in_pool"] = rel["pos"].notna()
rel["loc_other"] = rel["rel_loc"] != rel["search_location_id"]
rel["mc_missed"] = [mc not in top3.get(q, set()) for q, mc in zip(rel["qid"], rel["rel_mc"])]
rel["known_text"] = rel["qtext"].isin(hist_texts)
rel["infm"] = norm_text(rel["search_infm_params_text"]) != ""
rel["one_word"] = rel["qtext"].str.split().str.len() == 1
cell = corpus.groupby(["item_microcat_id", "item_location_id"]).size()
rel["rel_cell"] = [cell.get((a, b), 0) for a, b in zip(rel["rel_mc"], rel["search_location_id"])]

miss = rel[rel["miss"]]
print(f"пар (запрос, релевантное): {len(rel)}, промахов {len(miss)} ({len(miss) / len(rel):.1%}), запросов с промахом {miss['qid'].nunique()}")
share = lambda col: pd.Series({"промахи": miss[col].mean(), "попадания": rel.loc[~rel["miss"], col].mean()})
print(pd.DataFrame({
    "не в пуле K=200": share("in_pool").rsub(1),
    "релевантное в чужой локации": share("loc_other"),
    "microcat не в топ-3 предсказанных": share("mc_missed"),
    "текст известен истории": share("known_text"),
    "непустые фильтры": share("infm"),
    "запрос из 1 слова": share("one_word"),
}).T.round(3).to_string())
inpool = miss[miss["in_pool"]]
print("\nпромахи в пуле: позиция у ранкера, квантили", inpool["pos"].quantile([.25, .5, .75]).to_dict())
print("лучший ранг канала у промахов в пуле (медиана):", {c: inpool[f"rank_{c}"].median() for c in CHANNELS})

# 50 случайных промахов для ручной разметки
rng = np.random.default_rng(SEED)
sample = miss.iloc[rng.choice(len(miss), 50, replace=False)].sort_values("qid")
titles = f[f["pos"] <= 10].merge(corpus[["item_id", "item_title_raw"]], on="item_id")
with open(ARTIFACTS / "errors_p3.txt", "w") as fh:
    for n, r in enumerate(sample.itertuples(), 1):
        t = titles[titles["qid"] == r.qid]
        ranks = {c: int(getattr(r, f"rank_{c}")) for c in CHANNELS if pd.notna(getattr(r, f"rank_{c}"))}
        fh.write(f"#{n} [{r.qid}] запрос: «{r.search_query}» | фильтры: «{r.search_infm_params_text}» | loc {r.search_location_id}\n")
        fh.write(f"   релевантное: «{r.rel_title}» | loc {r.rel_loc}{' (ЧУЖАЯ)' if r.loc_other else ''} | mc {r.rel_mc}{' (не в топ-3)' if r.mc_missed else ''} "
                 f"| ячейка {r.rel_cell} | в пуле: {r.in_pool} поз. {r.pos} | ранги каналов {ranks}\n")
        fh.write("   топ-10: " + " ‖ ".join(f"{a} [{'=' if b == r.search_location_id else b}]" for a, b in zip(t['item_title_raw'], t['item_location_id'])) + "\n\n")
print("\nпримеры: artifacts/errors_p3.txt")

# --- Оценка пользы двух исправлений ---
TEXT = ["bm25_desc", "bm25_params", "bm25_title", "dense"]
f["text_support"] = (f[[f"rank_{c}" for c in TEXT]] <= meta["k"]).any(axis=1)
top = f[f["pos"] <= 50]
print(f"\nдоля ответа без поддержки текстовых каналов (нет в топ-{meta['k']} BM25/dense): {1 - top['text_support'].mean():.3f}")
nts = top.groupby("qid")["text_support"].apply(lambda s: (~s).sum())
# Верхняя оценка для признака «текстовое сходство для всех кандидатов пула»: промах в пуле
# с текстовой поддержкой, и его позиция ≤ 50 + число объявлений без текста выше него
mi = miss[miss["in_pool"]].merge(f[["qid", "item_id", "text_support"]], on=["qid", "item_id"])
fixable = mi[mi["text_support"] & (mi["pos"] <= 50 + mi["qid"].map(nts).fillna(0))]
print(f"промахов, которые поднялись бы в топ-50, если убрать объявления без текста: {len(fixable)} из {len(miss)} "
      f"(верхняя оценка +{len(fixable) / len(rel):.3f} к recall по парам)")

# Опечатки: доля запросов со словом (≥4 букв), которого нет в словаре корпуса
import re
vocab = set(re.findall(r"\w{4,}", " ".join(norm_text(corpus["item_title_raw"] + " " + corpus["item_infm_params_text"] + " "
                                                     + corpus["item_description_raw"].fillna("").str.slice(0, 1000)))))
oov = rel["qtext"].map(lambda t: any(w not in vocab for w in re.findall(r"\w{4,}", t)))
rel["oov"] = oov
print(f"запрос со словом вне словаря корпуса: промахи {rel.loc[rel['miss'], 'oov'].mean():.3f}, попадания {rel.loc[~rel['miss'], 'oov'].mean():.3f}")
print("примеры таких слов у промахов:", sorted({w for t in rel.loc[rel["miss"] & rel["oov"], "qtext"] for w in re.findall(r"\w{4,}", t) if w not in vocab})[:25])

# --- Оценка пользы канала «dense рядом с локацией запроса» ---
from src.channels.dense import _encode, encode_items
from src.data import query_text
ids, emb = encode_items(corpus)
pos_of = pd.Series(np.arange(len(ids)), index=ids)
cen = corpus.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
lat = corpus.set_index("item_id").loc[ids, "item_latitude"].to_numpy(np.float32)
lon = corpus.set_index("item_id").loc[ids, "item_longitude"].to_numpy(np.float32)
nip = miss[~miss["in_pool"]].copy()
qe = pd.Series(list(_encode(("query: " + query_text(p3.set_index("qid").loc[nip["qid"].unique()].reset_index())).tolist())),
               index=nip["qid"].unique())
rows = []
for r in nip.itertuples():
    c = cen.loc[r.search_location_id] if r.search_location_id in cen.index else None
    if c is None:
        rows.append((np.nan, np.nan, np.nan)); continue
    d = haversine_km(c["item_latitude"], c["item_longitude"], lat, lon)
    sim = emb @ qe[r.qid]
    j = pos_of[r.item_id]
    out = []
    for R in (30, 100):
        near = d <= R
        out.append(int((sim[near] > sim[j]).sum()) + 1 if d[j] <= R else np.nan)
    rows.append((d[j], *out))
nip[["rel_dist", "rank_near30", "rank_near100"]] = rows
print(f"\nпромахи вне пула: {len(nip)}; расстояние релевантного до центра локации запроса, квантили:",
      nip["rel_dist"].quantile([.25, .5, .75]).round(1).to_dict())
for R in (30, 100):
    col = f"rank_near{R}"
    print(f"  радиус {R} км: релевантное внутри {nip[col].notna().mean():.3f}; в топ-100 dense внутри радиуса {(nip[col] <= 100).mean():.3f} "
          f"(+{(nip[col] <= 100).sum()} пар в пул, {(nip[col] <= 100).sum() / len(rel):.3f} от всех пар)")
