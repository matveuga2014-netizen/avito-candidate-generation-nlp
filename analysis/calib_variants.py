"""Варианты схемы: сколько даёт бейзлайн при разной истории P1 (веса по локации benchmark).
Запуск: PYTHONPATH=. .venv/bin/python analysis/calib_variants.py"""
import numpy as np, pandas as pd
from src.data import load
from src.split import load_split, local_corpus
from src.baseline import popular_in_location
from src.metrics import recall_per_query

sp = load_split(); lc = local_corpus(sp)
p1 = sp[sp.part == "P1"]; p3 = sp[sp.part == "P3"]; hold = sp[sp.part != "P1"]
bq = load("benchmark_queries")
bl = bq.search_location_id.value_counts(normalize=True); pl = p3.search_location_id.value_counts(normalize=True)
w = p3.search_location_id.map(bl / pl).fillna(0).to_numpy()
truth = {q: set(i) for q, i in zip(p3.qid, p3["items"])}

# Кто в P1 выбрал релевантное P3: сколько таких экземпляров, та же ли локация, пересекается ли текст
e1 = p1[["qid", "qtext", "search_location_id", "items"]].explode("items")
e3 = p3[["qid", "qtext", "search_location_id", "items"]].explode("items")
m = e3.merge(e1, on="items", suffixes=("_3", "_1"))
m["same_loc"] = m.search_location_id_3 == m.search_location_id_1
tok = lambda s: set(s.split())
m["word_overlap"] = [len(tok(a) & tok(b)) > 0 for a, b in zip(m.qtext_3, m.qtext_1)]
print("пары (релевантное P3, экземпляр P1 с тем же объявлением):", len(m))
print("  та же локация:", round(m.same_loc.mean(), 4), "| есть общее слово в тексте:", round(m.word_overlap.mean(), 4))
per_item = m.groupby("items").qid_1.nunique()
print("  экземпляров P1 на такое объявление:", per_item.describe(percentiles=[.5, .9]).round(2).to_dict())

def run(name, hist):
    pred = popular_in_location(p3, hist, lc)
    r = recall_per_query(pred, truth).reindex(p3.qid).to_numpy()
    hi = set(hist["items"].explode())
    rel_in = p3["items"].map(lambda s: any(i in hi for i in s)).to_numpy()
    known = p3.qtext.isin(set(hist.qtext)).mean()
    print(f"{name:48s} P1={len(hist):6d} известн.={known:.3f} рел.в ист.={rel_in.mean():.3f} (взв. {np.average(rel_in, weights=w):.3f}) "
          f"recall={r.mean():.4f} взв.={np.average(r, weights=w):.4f}")

run("V0 текущая схема", p1)
hold_items = set(hold["items"].explode())
hit = p1["items"].map(lambda s: any(i in hold_items for i in s))
run("V1 убрать из P1 всё с объявлениями P2/P3", p1[~hit])
hold_il = set(zip(hold.explode("items")["items"], hold.explode("items").search_location_id))
hit2 = p1.explode("items").pipe(lambda d: pd.Series([(i, l) in hold_il for i, l in zip(d["items"], d.search_location_id)], index=d.index)).groupby(level=0).any()
run("V2 убрать из P1 то же объявление + та же локация", p1[~hit2.reindex(p1.index).to_numpy()])

# V3: для случайной доли p объявлений P2/P3 убираем из P1 все экземпляры с ними
print()
for p in (0.25, 0.30, 0.35):
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        hi = sorted(hold_items)
        drop = set(np.array(hi)[rng.random(len(hi)) < p])
        hit3 = p1["items"].map(lambda s: any(i in drop for i in s))
        run(f"V3 p={p} seed={seed}", p1[~hit3])
