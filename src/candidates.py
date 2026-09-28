"""Единая сборка кандидатов для P2, P3 и benchmark: одна логика для проверки и для ответа, без отдельных веток.

build_candidates(queries, history, corpus): различаются только аргументы.
Каналы делятся на две части: не зависящие от dense-модели (BM25, логи, приоры,
популярное) и dense (глобальный и в радиусе от локации запроса). Так смена
модели эмбеддингов пересчитывает только вторую часть (cached_candidates).
"""
import pandas as pd

from src.channels.bm25 import bm25_channels
from src.channels.dense import MODEL, dense_channel, dense_local_channel
from src.channels.logs import history_channels
from src.data import norm_text

K_POOL = 500
K_RRF = 60
N_ANSWER = 50
# Подобраны на P2 по приросту потолка пула:
# 50 км лучше 10/30/100 при любом N; топ-100 даёт +2.7 п.п. за +55 кандидатов на запрос.
LOCAL_RADIUS_KM = 50
LOCAL_N = 100
DENSE_CHANNELS = ("dense", "dense_local")


def _long(ch: dict) -> pd.DataFrame:
    return pd.concat([d.assign(channel=n) for n, d in ch.items()], ignore_index=True)


def base_candidates(queries, history, corpus, k=K_POOL) -> pd.DataFrame:
    queries = queries.assign(qtext=norm_text(queries["search_query"]))
    return _long({**bm25_channels(queries, corpus, k), **history_channels(queries, history, corpus, k)})


def dense_candidates(queries, corpus, k=K_POOL, model=MODEL, name="e5small") -> pd.DataFrame:
    return _long({"dense": dense_channel(queries, corpus, k, model, name),
                  "dense_local": dense_local_channel(queries, corpus, LOCAL_RADIUS_KM, LOCAL_N, model, name)})


def extra_channel(ch: str, name: str) -> str:
    """Каналы дополнительной dense-модели: dense_<имя>, dense_local_<имя>.
    Основная модель сохраняет имена dense / dense_local (совместимость с отправкой №4)."""
    return f"{ch}_{name}"


def _rename_extra(d: pd.DataFrame, name: str) -> pd.DataFrame:
    return d.assign(channel=d["channel"].map(lambda ch: extra_channel(ch, name)))


def build_candidates(queries, history, corpus, k=K_POOL, model=MODEL, name="e5small", extra=()) -> pd.DataFrame:
    """Длинная таблица qid, item_id, score, rank, channel. extra — ((имя, модель), ...) доп. dense-моделей."""
    parts = [base_candidates(queries, history, corpus, k), dense_candidates(queries, corpus, k, model, name)]
    parts += [_rename_extra(dense_candidates(queries, corpus, k, m, n), n) for n, m in extra]
    return pd.concat(parts, ignore_index=True)


def _key(params: dict) -> str:
    """Короткий хэш параметров, от которых зависит кэш: при их изменении кэш не подхватится молча."""
    import hashlib
    import json
    return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _ids_hash(s: pd.Series) -> str:
    import hashlib
    return hashlib.md5("\n".join(sorted(s.astype(str))).encode()).hexdigest()


def cache_paths(part, queries, history, corpus, model=MODEL, name="e5small"):
    """Пути кэша кандидатов. Ключ = хэш всего, от чего они зависят, чтобы кэш не
    подхватывался молча после смены параметров. База: запросы, история (через неё
    THIN_SHARE), корпус, K_POOL. Dense: запросы, корпус, K_POOL, LOCAL_RADIUS_KM,
    LOCAL_N, модель."""
    from src.split import ARTIFACTS, THIN_SHARE
    common = {"queries": _ids_hash(queries["qid"]), "corpus": _ids_hash(corpus["item_id"]), "k_pool": K_POOL}
    base_key = _key({**common, "history": _ids_hash(history["qid"]), "thin_share": THIN_SHARE})
    dense_key = _key({**common, "radius": LOCAL_RADIUS_KM, "local_n": LOCAL_N, "model": model, "name": name})
    return ARTIFACTS / f"cands_{part}_base_{base_key}.parquet", ARTIFACTS / f"cands_{part}_{name}_{dense_key}.parquet"


def cached_candidates(part, queries, history, corpus, model=MODEL, name="e5small", extra=()) -> pd.DataFrame:
    """build_candidates для P2/P3 с кэшем в artifacts/: база общая, dense на каждую модель
    (имя и путь модели входят в ключ кэша)."""
    base_p, dense_p = cache_paths(part, queries, history, corpus, model, name)
    if not base_p.exists():
        base_candidates(queries, history, corpus).to_parquet(base_p, index=False)

    def dense(p, m, n):
        if not p.exists():
            dense_candidates(queries, corpus, model=m, name=n).to_parquet(p, index=False)
        return pd.read_parquet(p)

    parts = [pd.read_parquet(base_p), dense(dense_p, model, name)]
    for n, m in extra:
        parts.append(_rename_extra(dense(cache_paths(part, queries, history, corpus, m, n)[1], m, n), n))
    return pd.concat(parts, ignore_index=True)


def rrf(cands: pd.DataFrame, weights: dict[str, float], n: int = N_ANSWER, k_rrf: int = K_RRF) -> dict[str, list[str]]:
    """score = Σ w_канала / (k_rrf + rank); ровно n уникальных id на запрос.

    pop_loc всегда даёт 50 объявлений на запрос, поэтому недобор добиваем из
    него по его же порядку: у каждого запроса есть фолбэк.
    """
    c = cands[cands["channel"].map(weights).fillna(0) > 0]
    s = (c["channel"].map(weights) / (k_rrf + c["rank"])).groupby([c["qid"], c["item_id"]]).sum().rename("s").reset_index()
    s = s.sort_values(["qid", "s", "item_id"], ascending=[True, False, True], kind="stable")
    top = s.groupby("qid").head(n).groupby("qid")["item_id"].agg(list).to_dict()
    fb = cands[cands["channel"] == "pop_loc"].sort_values(["qid", "rank"]).groupby("qid")["item_id"].agg(list).to_dict()
    out = {}
    for q, lst in fb.items():
        cur = top.get(q, [])
        seen = set(cur)
        out[q] = cur + [i for i in lst if i not in seen][: n - len(cur)]
        assert len(out[q]) == n == len(set(out[q])), q
    return out
