"""Одна команда от данных до answer.csv.

Первый этап: те же build_candidates, что для P2/P3, и CatBoost-ранкер варианта VARIANT,
обученный на P2 с историей P1 (src/ranker.py), не переобучается. Модель эмбеддингов и
набор каналов/признаков берутся из метаданных ранкера.
Второй этап (STAGE2_TAG): топ-150 или топ-300 первого этапа -> оценки cross-encoder из файла
(выход notebooks/kaggle_3_score_reranker.ipynb, проверяется, что они посчитаны именно для этих пар) ->
второй ранкер, обученный на P2 по OOF-оценкам первого этапа (src/stage2.py train2) -> топ-50.
STAGE2_TAG = "zeroshot_d300" — топ-300 (отправка №7), "zeroshot" — топ-150 (№6), None — ответ первого этапа (№5).
На benchmark история = весь train, корпус = benchmark_items; признаки строит тот же код, что при
отборе на P2/P3, различаются только аргументы.
"""
import json
import sys

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker

from src.candidates import N_ANSWER, build_candidates
from src.data import load
from src.ranker import build_features, meta_kwargs, predict_top
from src import stage2
from src.split import ARTIFACTS, load_split
from src.validate_answer import validate

VARIANT = "e5baseft_fix_bge_nopriors"
STAGE2_TAG = "zeroshot_d300"   # "zeroshot" — топ-150 (отправка №6); None — только первый этап (отправка №5)


def load_ranker(tag: str = VARIANT):
    """(модель, meta, аргументы признаков). tag "sub3" — ранкер отправки №3 (старые имена файлов)."""
    model_p, meta_p = (("ranker.cbm", "ranker_meta.json") if tag == "sub3"
                       else (f"ranker_{tag}.cbm", f"ranker_meta_{tag}.json"))
    meta = json.loads((ARTIFACTS / meta_p).read_text())
    m = CatBoostClassifier() if meta["loss"] == "Logloss" else CatBoostRanker()
    m.load_model(str(ARTIFACTS / model_p))
    return m, meta, meta_kwargs(meta)


def main(out: str = "answer.csv") -> None:
    sp = load_split()
    items = load("benchmark_items")
    bq = load("benchmark_queries").rename(columns={"query_id": "qid"})
    weights = json.loads((ARTIFACTS / "rrf_weights.json").read_text())
    m, meta, kw = load_ranker()

    cands = build_candidates(bq, sp, items, model=kw["model"], name=kw["name"], extra=kw["extra"])
    feats = build_features(cands, bq, sp, items, weights, k=meta["k"], **kw)
    if STAGE2_TAG is None:
        pred = predict_top(m, feats, meta["features"])
    else:
        m2meta = json.loads((stage2.STAGE2 / f"ranker2_{STAGE2_TAG}_meta.json").read_text())
        first = stage2._top(feats, stage2._scores(m, meta, feats), m2meta.get("top", 150))   # топ первого этапа
        assert m2meta["first"]["tag"] == VARIANT, "второй ранкер обучен на другом первом этапе"
        m2 = CatBoostClassifier() if m2meta["loss"] == "Logloss" else CatBoostRanker()
        m2.load_model(str(stage2.STAGE2 / f"ranker2_{STAGE2_TAG}.cbm"))
        dirs = {k: [ARTIFACTS.parent / x for x in v] if isinstance(v, list) else ARTIFACTS.parent / v
                for k, v in m2meta["ce_dirs"].items()}
        d = stage2.stage2_dataset(first, stage2.load_ce("bench", first, dirs))
        pred = predict_top(m2, d, m2meta["features"])
    assert all(len(pred[q]) == N_ANSWER == len(set(pred[q])) for q in bq["qid"]), "меньше 50 уникальных id"
    # Порядок строк как в benchmark_queries: ответ детерминирован и сравним по md5
    pd.DataFrame({"query_id": bq["qid"], "answer": [" ".join(pred[q]) for q in bq["qid"]]}).to_csv(out, index=False)
    validate(out, set(bq["qid"]), set(items["item_id"]))
    print(f"{out}: записан и проверен ({VARIANT}{' + второй этап ' + STAGE2_TAG if STAGE2_TAG else ''})")


if __name__ == "__main__":
    main(*sys.argv[1:])
