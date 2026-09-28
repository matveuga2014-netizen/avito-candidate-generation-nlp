"""Сетка dense {e5-base, bge-m3, обе} × priors {есть, нет}: 6 вариантов.

Отбор строго на части выбора P2 полной процедурой (src.ranker.train_variant,
eval_p3=False). Уже обученные тем же кодом варианты (метаданные с p2_select)
не переобучаются. На P3: лучший вариант (парная разница к e5baseft_fix, потолок
пула), а также лучший с priors и лучший без них — цена выбора по priors.

Перед запуском: artifacts/bge-m3-ft (веса), artifacts/emb_bgem3ft.npz (эмбеддинги корпуса с Kaggle).
Запуск: PYTHONPATH=. .venv/bin/python analysis/dense_grid.py
"""
import json

import pandas as pd

from src.metrics import paired_bootstrap
from src.ranker import p3_recall, part_data, train_variant
from src.split import ARTIFACTS, eval_history, load_split, local_corpus

E5 = ("e5baseft", "artifacts/e5-base-ft")
BGE = ("bgem3ft", "artifacts/bge-m3-ft")
DENSE = {"e5": (E5, ()), "bge": (BGE, ()), "обе": (E5, (BGE,))}
PRIORS = {"есть": (), "нет": ("priors",)}


def tag_of(primary, extra, drop):
    return f"{primary[0]}_fix{'_bge' if extra else ''}{'_nopriors' if drop else ''}"


rows = []
for dname, (primary, extra) in DENSE.items():
    for pname, drop in PRIORS.items():
        tag = tag_of(primary, extra, drop)
        mp = ARTIFACTS / f"ranker_meta_{tag}.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        if "p2_select" not in meta:
            meta = train_variant(primary[0], primary[1], True, drop_channels=drop, extra=extra,
                                 suffix=tag.removeprefix(f"{primary[0]}_fix"), eval_p3=False)
        rows.append({"dense": dname, "priors": pname, "тег": tag, "P2-выбор": meta["p2_select"],
                     "потери": meta["loss"], "деревьев": meta["iterations"]})
        print(rows[-1], flush=True)
tab = pd.DataFrame(rows).sort_values("P2-выбор", ascending=False)
print(tab.round(4).to_string(index=False))

sp = load_split(); corpus = local_corpus(sp); hist = eval_history(sp)
wts = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
_, _, w3 = part_data(sp, "P3")
base = pd.read_parquet(ARTIFACTS / "p3_recall_e5baseft_fix.parquet")["recall"]


def p3(tag):
    path = ARTIFACTS / f"p3_recall_{tag}.parquet"
    if not path.exists():
        from catboost import CatBoostClassifier, CatBoostRanker
        meta = json.loads((ARTIFACTS / f"ranker_meta_{tag}.json").read_text())
        m = CatBoostClassifier() if meta["loss"] == "Logloss" else CatBoostRanker()
        m.load_model(str(ARTIFACTS / f"ranker_{tag}.cbm"))
        p3_recall(m, meta, sp, corpus, hist, wts).rename("recall").to_frame().to_parquet(path)
    r = pd.read_parquet(path)["recall"].reindex(base.index)
    wv = w3.reindex(r.index).to_numpy()
    d, lo, hi = paired_bootstrap(r, base, wv)
    return f"P3 {(r * wv).sum() / wv.sum():.4f} | к e5baseft_fix (0.849) {100 * d:+.2f} п.п. [{100 * lo:+.2f}, {100 * hi:+.2f}]"


best = tab.iloc[0]["тег"]
print(f"лучший по P2: {best} — {p3(best)}")
for pname in PRIORS:
    t = tab[tab["priors"] == pname].iloc[0]["тег"]
    print(f"лучший с priors = {pname}: {t} — {p3(t)}")
