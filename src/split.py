"""Разбиение train на P1 (история) / P2 (ранкер) / P3 (отложенная проверка).

Схема повторяет устройство benchmark (см. README и notebooks/01_eda.ipynb):
случайный уникальный текст -> один его случайный экземпляр. Остальные экземпляры
того же текста остаются в истории, поэтому доля «известных» запросов получается
как в benchmark без ручной подгонки.

Запуск: python -m src.split  (строит artifacts/split.parquet, если его нет,
и печатает самопроверку B).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import QUERY_KEY, build_instances, load, norm_text

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
SPLIT_PATH = ARTIFACTS / "split.parquet"
SEED = 42
N_P2, N_P3 = 8000, 5000
# Доля объявлений P2/P3, для которых экземпляры с ними убираются из истории.
# Выбрана по отправке №1 (см. README, «Калибровка»): без прореживания P3 давал
# 0.149 против 0.0797 на платформе, правило «то же объявление + та же локация»
# без параметра дало 0.031 (вне 0.07–0.09), доля 0.3 даёт ≈0.08.
THIN_SHARE = 0.3


def assign_parts(inst: pd.DataFrame, n_p2: int = N_P2, n_p3: int = N_P3, seed: int = SEED) -> pd.Series:
    """Возвращает часть ("P1"/"P2"/"P3") для каждого экземпляра.

    Всё сортируется перед случайным выбором, чтобы результат зависел только
    от seed и содержимого, а не от порядка строк во входе.
    """
    rng = np.random.default_rng(seed)
    texts = np.sort(inst["qtext"].unique())
    order = rng.permutation(len(texts))
    text_part = {t: "P2" for t in texts[order[:n_p2]]}
    text_part.update({t: "P3" for t in texts[order[n_p2:n_p2 + n_p3]]})

    # Один случайный экземпляр на выбранный текст: случайный ключ, затем первый в группе.
    cand = inst.loc[inst["qtext"].isin(text_part), ["qid", "qtext"]].sort_values("qid")
    cand["r"] = rng.random(len(cand))
    picked = cand.sort_values(["qtext", "r"]).drop_duplicates("qtext")

    part = pd.Series("P1", index=inst.index)
    part[picked.index] = picked["qtext"].map(text_part)
    return part


def thin_mask(sp: pd.DataFrame, share: float = THIN_SHARE, seed: int = SEED) -> pd.Series:
    """True для экземпляров P1, которые убираются из истории при оценке P2/P3.

    Для случайной доли share объявлений, релевантных в P2/P3, из истории
    убираются все экземпляры P1 с ними. Без этого 41% (взвешенно) релевантных
    P3 уже есть в истории, а на benchmark, судя по отправке №1, около 29%.
    Финальный прогон на benchmark идёт с полным train, без прореживания.
    """
    hold_items = np.array(sorted(set(sp.loc[sp["part"] != "P1", "items"].explode())))
    drop = set(hold_items[np.random.default_rng(seed).random(len(hold_items)) < share])
    return (sp["part"] == "P1") & sp["items"].map(lambda s: any(i in drop for i in s))


def make_split() -> pd.DataFrame:
    train = load("train", QUERY_KEY + ["item_id"])
    inst = build_instances(train)
    inst["part"] = assign_parts(inst)
    inst = inst.sort_values("qid").reset_index(drop=True)
    inst["thinned"] = thin_mask(inst)
    return inst


def load_split() -> pd.DataFrame:
    """Разбиение из файла; строится один раз, дальше только загружается."""
    if not SPLIT_PATH.exists():
        ARTIFACTS.mkdir(exist_ok=True)
        make_split().to_parquet(SPLIT_PATH, index=False)
    sp = pd.read_parquet(SPLIT_PATH)
    assert sp["qid"].map(type).eq(str).all()
    sp["items"] = sp["items"].map(list)
    return sp


def eval_history(sp: pd.DataFrame) -> pd.DataFrame:
    """История для P2/P3: P1 без прореженных экземпляров."""
    return sp[(sp["part"] == "P1") & ~sp["thinned"]]


def location_weights(locs: pd.Series) -> pd.Series:
    """Вес запроса = доля его локации в benchmark / доля в этой выборке.

    P3 (и P2) сильнее размазаны по маленьким локациям, где бейзлайны находят
    ответ легче; веса выравнивают распределение городов под benchmark.
    Берутся только локации запросов benchmark, без разметки. Локации, которых
    нет в benchmark, получают вес 0.
    """
    bench = load("benchmark_queries", ["search_location_id"])["search_location_id"].value_counts(normalize=True)
    own = locs.value_counts(normalize=True)
    return locs.map(bench / own).fillna(0.0)


def local_corpus(split: pd.DataFrame) -> pd.DataFrame:
    """benchmark_items + релевантные объявления P2/P3, которых в нём нет.

    Объявления train почти не лежат в корпусе (5.3% по EDA), без добавления
    релевантных P2/P3 было бы нечего находить. id настоящие: при совпадении id
    контент в train и корпусе одинаков (EDA, раздел 3).
    """
    items = load("benchmark_items")
    need = set(split.loc[split["part"] != "P1", "items"].explode()) - set(items["item_id"])
    added = load("train", list(items.columns), filters=[("item_id", "in", sorted(need))])
    added = added.drop_duplicates("item_id")
    assert len(added) == len(need)
    corpus = pd.concat([items, added], ignore_index=True)
    assert corpus["item_id"].is_unique
    return corpus


# ---------- самопроверка B ----------

def _item_features(d: pd.DataFrame) -> pd.DataFrame:
    """Простые признаки контента для adversarial validation (без id и категорий)."""
    return pd.DataFrame({
        "no_rating": d["item_rating"].isna().astype(int),
        "rating": d["item_rating"].fillna(-1),
        "log_reviews": np.log1p(d["item_rating_reviews_count"].fillna(0)),
        "log_price": np.log1p(d["item_price"].clip(lower=0)),
        "title_len": d["item_title_raw"].str.len(),
        "params_len": d["item_infm_params_text"].str.len(),
        "desc_len": d["item_description_raw"].fillna("").str.len(),
        "phone_hidden": d["item_is_phone_hidden"].astype(int),
        "msg_forbidden": d["item_is_message_forbidden"].astype(int),
    })


def adversarial_auc(pos: pd.DataFrame, neg: pd.DataFrame) -> tuple[float, pd.Series]:
    """ROC AUC классификатора «вставленное / из корпуса» (5-fold CV) и AUC каждого признака.

    AUC ≈ 0.5: не различимы, локальная оценка честная. Заметно выше:
    ранкер может учиться находить вставленные объявления.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    X = pd.concat([_item_features(pos), _item_features(neg)], ignore_index=True)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(HistGradientBoostingClassifier(random_state=SEED), X, y, cv=cv, method="predict_proba")[:, 1]
    # AUC по одному признаку: max(auc, 1-auc), направление не важно
    per_feat = X.apply(lambda c: max(roc_auc_score(y, c), 1 - roc_auc_score(y, c))).sort_values(ascending=False)
    return roc_auc_score(y, proba), per_feat


def check_split(sp: pd.DataFrame) -> None:
    parts = {p: sp[sp["part"] == p] for p in ("P1", "P2", "P3")}
    print("экземпляров:", {p: len(d) for p, d in parts.items()},
          f"| P1 = {len(parts['P1']) / len(sp):.1%}")
    assert len(parts["P2"]) == N_P2 and len(parts["P3"]) == N_P3
    assert parts["P2"]["qtext"].is_unique and parts["P3"]["qtext"].is_unique, "в P2/P3 больше одного экземпляра на текст"
    assert not set(parts["P2"]["qtext"]) & set(parts["P3"]["qtext"]), "тексты P2 и P3 пересекаются"
    assert not set(parts["P1"]["qid"]) & set(sp.loc[sp["part"] != "P1", "qid"])

    # «Известные» запросы: текст есть в истории. Benchmark: текст в train.
    bq = load("benchmark_queries")
    train_texts = set(sp["qtext"])
    hist = eval_history(sp)
    print(f"история для P2/P3: {len(hist)} экземпляров, прорежено {int(sp['thinned'].sum())}")
    assert not sp.loc[sp["part"] != "P1", "thinned"].any()
    p1_texts = set(hist["qtext"])
    print("доля известных текстов: P2 {:.4f} | P3 {:.4f} | benchmark {:.4f}".format(
        parts["P2"]["qtext"].isin(p1_texts).mean(), parts["P3"]["qtext"].isin(p1_texts).mean(),
        norm_text(bq["search_query"]).isin(train_texts).mean()))
    # Точное совпадение экземпляра: в P2/P3 невозможно по построению (ограничение, см. README)
    print("точное совпадение экземпляра с историей: P3 0 по построению | benchmark {:.4f}".format(
        bq[QUERY_KEY].merge(sp[QUERY_KEY].drop_duplicates(), how="inner").shape[0] / len(bq)))
    infm = lambda d: (norm_text(d["search_infm_params_text"]) != "").mean()
    print("доля непустого infm_params: P3 {:.4f} | benchmark {:.4f}".format(infm(parts["P3"]), infm(bq)))

    # Локальный корпус: все релевантные P2/P3 в нём, размер того же порядка
    corpus = local_corpus(sp)
    items_ids = set(load("benchmark_items", ["item_id"])["item_id"])
    rel = set(sp.loc[sp["part"] != "P1", "items"].explode())
    assert rel <= set(corpus["item_id"]), "не все релевантные P2/P3 в локальном корпусе"
    inserted = corpus[~corpus["item_id"].isin(items_ids)]
    print(f"локальный корпус: {len(corpus)} (benchmark_items {len(items_ids)}, вставлено {len(inserted)}, "
          f"{len(inserted) / len(corpus):.1%})")

    # Доля объявлений из истории в корпусе: локально история = P1, для benchmark = весь train
    p1_items = set(hist["items"].explode())
    all_items = set(sp["items"].explode())
    print("доля корпуса из истории: локальный (P1 прореж.) {:.4f} | benchmark_items (train) {:.4f}".format(
        corpus["item_id"].isin(p1_items).mean(), len(items_ids & all_items) / len(items_ids)))

    # Adversarial validation: вставленные против объявлений корпуса, встречающихся в train
    ref = corpus[corpus["item_id"].isin(items_ids & all_items)]
    auc, per_feat = adversarial_auc(inserted, ref)
    # Веса по локации и доля релевантных P3 в истории (цель по отправке №1 ≈ 0.29 взвешенно)
    w = location_weights(parts["P3"]["search_location_id"])
    print(f"веса P3: ESS {w.sum() ** 2 / (w ** 2).sum():.0f} из {len(w)}, макс {w.max():.2f}")
    rel_in = parts["P3"]["items"].map(lambda s: any(i in p1_items for i in s))
    print(f"доля P3 с релевантным в истории: {rel_in.mean():.4f}, взвешенно {np.average(rel_in, weights=w):.4f}")
    print(f"adversarial AUC (вставленные {len(inserted)} vs корпус∩train {len(ref)}): {auc:.4f}")
    print("AUC по одному признаку:", per_feat.round(4).to_dict())


if __name__ == "__main__":
    sp = load_split()
    check_split(sp)
