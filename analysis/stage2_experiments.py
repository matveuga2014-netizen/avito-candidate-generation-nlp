"""Улучшения второго ранкера: отбор на части выбора P2, одна проверка на P3 (парная разница к базе 0.889).

Варианты: взаимодействия CE × loc_match / dist_km; нормировка CE внутри запроса (разность со средним топ-5,
доля кандидатов выше порога — порог из трёх квантилей оценок P2 выбирается на P2); ансамбль (среднее рангов
Logloss и YetiRank, 3 seed); совпадение фильтров запроса и объявления; комбинации. Кривая обучения базы:
25/50/100% P2. Результаты — в artifacts/stage2/experiments.json и в лог.
Запуск: PYTHONPATH=. .venv/bin/python analysis/stage2_experiments.py
"""
import json

import numpy as np
import pandas as pd

from src.data import load
from src.metrics import paired_bootstrap
from src.split import ARTIFACTS, load_split, local_corpus, location_weights
from src.stage2 import CE_DIRS, STAGE2, add_ce_extra, add_filter_features, load_ce, stage2_dataset, train_stage2

sp = load_split()
first = {p: pd.read_parquet(STAGE2 / f"first_{p}.parquet") for p in ("P2", "P3")}
base_d = {p: stage2_dataset(first[p], load_ce(p, first[p], CE_DIRS)) for p in ("P2", "P3")}
truth = {p: dict(zip(sp.loc[sp.part == p, "qid"], map(set, sp.loc[sp.part == p, "items"]))) for p in ("P2", "P3")}
w = {p: pd.Series(location_weights(sp.loc[sp.part == p, "search_location_id"]).to_numpy(), index=sp.loc[sp.part == p, "qid"])
     for p in ("P2", "P3")}
q_params = sp.set_index("qid")["search_infm_params_text"]
i_params = local_corpus(sp).set_index("item_id")["item_infm_params_text"]
flt = {p: add_filter_features(base_d[p], q_params, i_params) for p in ("P2", "P3")}
base_r3 = pd.read_parquet(ARTIFACTS / "p3_recall_stage2_zeroshot.parquet")["recall"]
results = []


def run(name, d, **kw):
    _, info, r3 = train_stage2(d["P2"], truth["P2"], w["P2"], d["P3"], truth["P3"], w["P3"], **kw)
    b = base_r3.reindex(r3.index)
    wv = w["P3"].reindex(r3.index).to_numpy()
    diff, lo, hi = paired_bootstrap(r3, b, wv)
    row = {"вариант": name, "P2 выбор": info["p2_select"], "P3": info["p3"], "к 0.889, п.п.": 100 * diff,
           "от": 100 * lo, "до": 100 * hi, "модель": info.get("loss"), "деревьев": info.get("iterations")}
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(results, open(STAGE2 / "experiments.json", "w"), ensure_ascii=False, indent=1)
    return row


def with_extra(src, groups, thr=None):
    return {p: add_ce_extra(src[p], groups, thr) for p in ("P2", "P3")}


run("база (bge, топ-150)", base_d)
run("+ CE × локация/расстояние", with_extra(base_d, ("inter",)))
thrs = np.quantile(base_d["P2"]["ce_zeroshot"], [0.90, 0.97, 0.99])
norm_rows = [run(f"+ нормировка CE, порог q{q}", with_extra(base_d, ("norm",), t)) for q, t in zip((90, 97, 99), thrs)]
best_thr = thrs[int(np.argmax([r["P2 выбор"] for r in norm_rows]))]
print("порог по P2:", best_thr, flush=True)
run("ансамбль 6 моделей", base_d, ensemble=True)
run("+ CE × локация + нормировка + ансамбль", with_extra(base_d, ("inter", "norm"), best_thr), ensemble=True)
run("+ фильтры", flt)
run("+ фильтры + CE × локация + нормировка", with_extra(flt, ("inter", "norm"), best_thr))
run("+ фильтры + CE × локация + нормировка + ансамбль", with_extra(flt, ("inter", "norm"), best_thr), ensemble=True)
for frac in (0.25, 0.5):
    run(f"кривая обучения: база на {int(frac * 100)}% P2", base_d, p2_frac=frac)
print(pd.DataFrame(results).round(4).to_string(index=False))
