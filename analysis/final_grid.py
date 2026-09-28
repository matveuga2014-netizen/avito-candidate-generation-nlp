"""Итоговая сетка второго этапа: глубина {150, 300} × признаки {база, нормировка CE q99 — лучший по P2
набор из analysis/stage2_experiments.py}; только bge-reranker-v2-m3 без дообучения. Отбор на части
выбора P2, лучший — на P3 с парной разницей к 0.889 и потолком топа; лог в artifacts/stage2/final_grid.json.
Запуск: PYTHONPATH=. .venv/bin/python analysis/final_grid.py
"""
import json

import numpy as np
import pandas as pd

from src.metrics import paired_bootstrap, recall_at_k
from src.split import ARTIFACTS, load_split, location_weights
from src.stage2 import STAGE2, add_ce_extra, load_ce, stage2_dataset, train_stage2

sp = load_split()
truth = {p: dict(zip(sp.loc[sp.part == p, "qid"], map(set, sp.loc[sp.part == p, "items"]))) for p in ("P2", "P3")}
w = {p: pd.Series(location_weights(sp.loc[sp.part == p, "search_location_id"]).to_numpy(), index=sp.loc[sp.part == p, "qid"])
     for p in ("P2", "P3")}
B150, BDEPTH = ARTIFACTS / "kaggle_output_B_zeroshot", ARTIFACTS / "kaggle_output_B_depth"
DEPTH = {150: ("", {"zeroshot": B150}), 300: ("_300", {"zeroshot": [B150, BDEPTH]})}
first150 = pd.read_parquet(STAGE2 / "first_P2.parquet")
d150 = stage2_dataset(first150, load_ce("P2", first150, DEPTH[150][1]))
THR = float(np.quantile(d150["ce_zeroshot"], 0.99))
base = pd.read_parquet(ARTIFACTS / "p3_recall_stage2_zeroshot.parquet")["recall"]
rows = []
for depth, (suf, dirs) in DEPTH.items():
    first = {p: pd.read_parquet(STAGE2 / f"first_{p}{suf}.parquet") for p in ("P2", "P3")}
    d = {p: stage2_dataset(first[p], load_ce(p, first[p], dirs)) for p in ("P2", "P3")}
    ceil = recall_at_k(first["P3"].sort_values(["qid", "first_rank"]).groupby("qid")["item_id"].agg(list).to_dict(),
                       truth["P3"], depth, w["P3"])
    for feat, dd in (("база", d), ("нормировка q99", {p: add_ce_extra(d[p], ("norm",), THR) for p in d})):
        _, info, r3 = train_stage2(dd["P2"], truth["P2"], w["P2"], dd["P3"], truth["P3"], w["P3"])
        wv = w["P3"].reindex(r3.index).to_numpy()
        diff, lo, hi = paired_bootstrap(r3, base.reindex(r3.index), wv)
        rows.append({"глубина": depth, "признаки": feat, "P2 выбор": info["p2_select"], "P3": info["p3"],
                     "к 0.889": 100 * diff, "от": 100 * lo, "до": 100 * hi, f"потолок": ceil, "модель": info["loss"]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        json.dump(rows, open(STAGE2 / "final_grid.json", "w"), ensure_ascii=False, indent=1)
tab = pd.DataFrame(rows).sort_values("P2 выбор", ascending=False)
print(tab.round(4).to_string(index=False))
print("лучший по P2:", tab.iloc[0][["глубина", "признаки"]].to_dict())
