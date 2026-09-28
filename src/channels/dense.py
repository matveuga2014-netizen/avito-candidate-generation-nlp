"""Dense-канал: multilingual-e5, точный поиск перемножением матриц.

Эмбеддинги корпуса кэшируются в artifacts/: локальный корпус содержит весь
benchmark_items, поэтому один кэш покрывает и валидацию, и benchmark.
"""
import os

import numpy as np
import pandas as pd

from src.data import haversine_km, item_dense_text, location_centroids, query_text
from src.split import ARTIFACTS

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # модель скачана один раз, дальше офлайн
MODEL = "intfloat/multilingual-e5-small"
MAX_LEN = 128
BATCH = 128


E5_PREFIXES = ("query: ", "passage: ")
NO_PREFIXES = ("", "")
# Формат входа задаётся явно для каждой модели (раньше — по подстроке «e5» в пути, что ломается
# при переименовании каталогов). Ключ — имя на HF или путь, как в метаданных ранкера.
MODEL_PREFIXES = {
    "intfloat/multilingual-e5-small": E5_PREFIXES,
    "intfloat/multilingual-e5-base": E5_PREFIXES,
    "artifacts/e5-base-ft": E5_PREFIXES,          # дообученная e5-base: префиксы обязательны
    "deepvk/USER-bge-m3": NO_PREFIXES,
    "artifacts/bge-m3-ft": NO_PREFIXES,           # дообученная USER-bge-m3: без префиксов
}


def prefixes(model: str) -> tuple[str, str]:
    """Префиксы запроса и объявления для модели; неизвестная модель — ошибка, а не догадка."""
    if model not in MODEL_PREFIXES:
        raise KeyError(f"неизвестна схема префиксов для модели {model!r}: добавьте её в MODEL_PREFIXES")
    return MODEL_PREFIXES[model]


def _model(model: str = MODEL):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(42)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(model, device=dev)
    m.max_seq_length = MAX_LEN
    return m


def encode_items(corpus: pd.DataFrame, name: str = "e5small", model: str = MODEL) -> tuple[np.ndarray, np.ndarray]:
    """(item_ids, эмбеддинги float32). Префикс объявления — prefixes(model)."""
    path = ARTIFACTS / f"emb_{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        # Эмбеддинги с Kaggle (bge-m3) лежат в float16 ради размера Output; считаем в float32
        ids, emb = z["ids"], z["emb"].astype(np.float32)
        # Кэш знает, какой моделью посчитан: другая модель под тем же именем — ошибка, а не тихое смешение
        if "model" in z.files and str(z["model"]) != model:
            raise ValueError(f"{path.name} посчитан моделью {z['model']}, запрошена {model}")
        missing = ~pd.Index(corpus["item_id"]).isin(ids)
        if not missing.any():
            return ids, emb
        # догружаем недостающие (другой корпус) и расширяем кэш
        extra = corpus[missing]
        e2 = _encode((prefixes(model)[1] + item_dense_text(extra)).tolist(), model)
        ids, emb = np.r_[ids, extra["item_id"].to_numpy()], np.vstack([emb, e2])
    else:
        ids = corpus["item_id"].to_numpy()
        emb = _encode((prefixes(model)[1] + item_dense_text(corpus)).tolist(), model)
    np.savez(path, ids=ids.astype(str), emb=emb, model=np.array(model))
    return ids, emb


def _encode(texts: list[str], model: str = MODEL) -> np.ndarray:
    return _model(model).encode(texts, batch_size=BATCH, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def corpus_embeddings(corpus: pd.DataFrame, name: str = "e5small", model: str = MODEL) -> tuple[np.ndarray, np.ndarray]:
    """Эмбеддинги ровно объявлений этого корпуса, в его порядке (кэш шире корпуса)."""
    ids, emb = encode_items(corpus, name, model)
    pos = pd.Series(np.arange(len(ids)), index=ids)[corpus["item_id"]].to_numpy()
    return ids[pos], emb[pos]


def _stable_top(qids, sim, ids, k):
    """Топ-k по строкам sim со стабильным порядком (score убыв., item_id)."""
    k = min(k, sim.shape[1])
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    out = []
    for j, row in enumerate(top):
        sc = sim[j, row]
        o = np.lexsort((ids[row], -sc))
        out.append(pd.DataFrame({"qid": qids[j], "item_id": ids[row][o], "score": sc[o], "rank": np.arange(1, k + 1)}))
    return out


def dense_local_channel(queries: pd.DataFrame, corpus: pd.DataFrame, radius_km: float, n: int,
                        model: str = MODEL, name: str = "e5small") -> pd.DataFrame:
    """Топ-n по косинусу среди объявлений в радиусе radius_km от центра локации запроса.

    Глобальный dense отдаёт топ-500 по всей России, и нужное объявление рядом
    с пользователем тонет среди одинаковых из других городов (34% промахов P3).
    Запросы с локацией без объявлений в корпусе кандидатов не получают.
    """
    ids, emb = corpus_embeddings(corpus, name, model)
    lat = corpus["item_latitude"].to_numpy(np.float64)
    lon = corpus["item_longitude"].to_numpy(np.float64)
    cen = location_centroids(corpus)
    q = _encode((prefixes(model)[0] + query_text(queries)).tolist(), model)
    qids = queries["qid"].to_numpy()
    out = []
    for loc, idx in sorted(queries.groupby("search_location_id").indices.items()):
        if loc not in cen.index:
            continue
        near = np.flatnonzero(haversine_km(cen.at[loc, "item_latitude"], cen.at[loc, "item_longitude"], lat, lon) <= radius_km)
        if len(near):
            out += _stable_top(qids[idx], q[idx] @ emb[near].T, ids[near], n)
    return pd.concat(out, ignore_index=True)


def dense_channel(queries: pd.DataFrame, corpus: pd.DataFrame, k: int = 500,
                  model: str = MODEL, name: str = "e5small") -> pd.DataFrame:
    """Топ-k по косинусу. Возвращает qid, item_id, score, rank.

    model — имя на HF или локальный путь (дообученная модель в artifacts/),
    name — имя кэша эмбеддингов корпуса.
    """
    ids, emb = corpus_embeddings(corpus, name, model)
    q = _encode((prefixes(model)[0] + query_text(queries)).tolist(), model)
    out = []
    qids = queries["qid"].to_numpy()
    for s0 in range(0, len(q), 1024):
        out += _stable_top(qids[s0:s0 + 1024], q[s0:s0 + 1024] @ emb.T, ids, k)
    return pd.concat(out, ignore_index=True)


def pair_cosine(queries: pd.DataFrame, corpus: pd.DataFrame, pairs: pd.DataFrame,
                model: str = MODEL, name: str = "e5small") -> np.ndarray:
    """Косинус запроса и объявления для заданных пар (qid, item_id), для всего пула."""
    ids, emb = corpus_embeddings(corpus, name, model)
    q = _encode((prefixes(model)[0] + query_text(queries)).tolist(), model)
    qi = pd.Series(np.arange(len(queries)), index=queries["qid"])[pairs["qid"]].to_numpy()
    ii = pd.Series(np.arange(len(ids)), index=ids)[pairs["item_id"]].to_numpy()
    out = np.empty(len(pairs), dtype=np.float32)
    for s0 in range(0, len(pairs), 500_000):  # кусками: пары × 384 float32 целиком не влезут
        sl = slice(s0, s0 + 500_000)
        out[sl] = np.einsum("ij,ij->i", q[qi[sl]], emb[ii[sl]])
    return out
