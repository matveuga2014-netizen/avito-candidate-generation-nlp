"""Разбор расхождения бейзлайна P3 0.149 и платформы 0.0797: сбой, сдвиг по городам, пересечение с историей.
Запуск: PYTHONPATH=. .venv/bin/python analysis/calib_baseline.py"""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
from src.data import load
from src.split import load_split, local_corpus
from src.baseline import popular_in_location
from src.metrics import recall_per_query

sp = load_split(); p1 = sp[sp.part == "P1"]; p3 = sp[sp.part == "P3"].copy()
items = load("benchmark_items"); lc = local_corpus(sp)
bq = load("benchmark_queries").rename(columns={"query_id": "qid"})
bench_ids = set(items.item_id)

def describe(name, q, hist, corpus):
    pred = popular_in_location(q, hist, corpus)
    pop = hist["items"].explode().value_counts()
    loc_of = dict(zip(corpus.item_id, corpus.item_location_id))
    n_in_loc = corpus.item_location_id.value_counts()
    hist_locs = set(hist.search_location_id)
    rows = []
    for qid, loc in zip(q.qid, q.search_location_id):
        ids = pred[qid]
        rows.append({"qid": qid, "loc": loc, "loc_in_hist": loc in hist_locs,
                     "n_corpus_loc": n_in_loc.get(loc, 0),
                     "from_loc": sum(loc_of[i] == loc for i in ids),
                     "pop_pos": sum(pop.get(i, 0) > 0 for i in ids)})
    d = pd.DataFrame(rows)
    print(f"\n== {name}: запросов {len(d)}")
    print("доля запросов с локацией в истории:", round(d.loc_in_hist.mean(), 4))
    print("id из локации / из фолбэка (доля):", round(d.from_loc.sum() / (50 * len(d)), 4), "/", round(1 - d.from_loc.sum() / (50 * len(d)), 4))
    print("id с популярностью > 0 в истории:", round(d.pop_pos.sum() / (50 * len(d)), 4))
    b = pd.cut(d.n_corpus_loc, [-1, 0, 49, 199, 999, 4999, 10**9], labels=["0", "1-49", "50-199", "200-999", "1k-5k", "5k+"])
    print("объявлений корпуса в локации запроса:", b.value_counts(normalize=True).sort_index().round(4).to_dict(), "| медиана", d.n_corpus_loc.median())
    return pred, d

pred3, d3 = describe("P3 (корпус локальный, история P1)", p3, p1, lc)
_, db = describe("benchmark (benchmark_items, история train)", bq, sp, items)

truth = {q: set(i) for q, i in zip(p3.qid, p3["items"])}
r = recall_per_query(pred3, truth).reindex(p3.qid).to_numpy()
p3["recall"] = r
print("\nRecall P3:", round(r.mean(), 4))

# --- сдвиг по городам: размер города = объявлений benchmark_items в локации (одинаково для обеих сторон)
size = items.item_location_id.value_counts()
bucket = lambda s: pd.cut(s.map(size).fillna(0), [-1, 0, 49, 199, 999, 4999, 10**9], labels=["0", "1-49", "50-199", "200-999", "1k-5k", "5k+"])
p3["b"] = bucket(p3.search_location_id); bq["b"] = bucket(bq.search_location_id)
tab = pd.DataFrame({"P3 доля": p3.b.value_counts(normalize=True), "bench доля": bq.b.value_counts(normalize=True),
                    "P3 recall": p3.groupby("b", observed=True).recall.mean()}).sort_index().round(4)
print("\nбакеты по размеру города (объявлений benchmark_items в локации):\n", tab)
w = p3.b.map(tab["bench доля"] / tab["P3 доля"]).astype(float)
print("P3 recall, веса по бакетам:", round(np.average(r, weights=w), 4))
pl = p3.search_location_id.value_counts(normalize=True); bl = bq.search_location_id.value_counts(normalize=True)
wl = p3.search_location_id.map(bl / pl).fillna(0)
print("P3 recall, веса по локации:", round(np.average(r, weights=wl), 4),
      "| покрыто массы benchmark:", round(bl[bl.index.isin(pl.index)].sum(), 4))

# --- пересечение с историей
p1_items = set(p1["items"].explode())
rel = p3["items"].explode()
print("\nдоля релевантных P3, встречающихся в P1:", round(rel.isin(p1_items).mean(), 4))
p3["rel_in_hist"] = p3["items"].map(lambda s: any(i in p1_items for i in s))
print(p3.groupby("rel_in_hist").recall.agg(["size", "mean"]).round(4))

# --- вставка: релевантное вставлено в корпус или уже было в benchmark_items
p3["inserted"] = p3["items"].map(lambda s: all(i not in bench_ids for i in s))
print("\nрелевантные вставлены (все) / уже в benchmark_items:")
print(p3.groupby("inserted").recall.agg(["size", "mean"]).round(4))
print(p3.groupby(["b", "inserted"], observed=True).recall.agg(["size", "mean"]).round(4).unstack())
p3.drop(columns=["items"]).to_parquet(f"artifacts/p3_calib.parquet")
