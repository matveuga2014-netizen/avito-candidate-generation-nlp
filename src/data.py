"""Загрузка данных, нормализация текста, ключ экземпляра запроса, хэш контента.

Одна точка правды для всего проекта: EDA, разбиение, каналы и make_answer.py
берут эти функции отсюда, чтобы предобработка везде совпадала.
"""
import hashlib
import re

import numpy as np
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# id всегда строки: приведение к числу теряет ведущие нули и ломает сдачу.
ID_COLS = ("item_id", "query_id")

# Экземпляр запроса = всё, что пользователь задал в поиске. Строки train
# с одинаковым ключом считаем одним запросом с несколькими релевантными.
QUERY_KEY = [
    "search_query",
    "search_location_id",
    "search_infm_params_text",
    "search_category",
    "search_is_delivery_search",
]

_WS = re.compile(r"\s+")


def load(name: str, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """Читает data/<name>.parquet, только нужные колонки (и строки, если задан filters).

    decimal128 (цена, координаты) переводим в float64: pandas иначе держит их
    как объекты Decimal, это медленно и много памяти.
    """
    table = pq.read_table(DATA_DIR / f"{name}.parquet", columns=columns, filters=filters)
    for i, field in enumerate(table.schema):
        if pa.types.is_decimal(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.float64()))
    df = table.to_pandas()
    for col in ID_COLS:
        if col in df.columns:
            assert df[col].notna().all(), f"NaN в {col}"
            assert df[col].map(type).eq(str).all(), f"{col} не строка"
    return df


def norm_text(s: pd.Series) -> pd.Series:
    """Единая нормализация текста: пропуск -> "", lower, ё -> е, схлопнутые пробелы."""
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(_WS, " ", regex=True)
        .str.strip()
    )


def query_key(df: pd.DataFrame) -> pd.Series:
    """Строковый ключ экземпляра запроса; текстовые поля нормализованы."""
    parts = [
        norm_text(df[c]) if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
        else df[c].astype(str)
        for c in QUERY_KEY
    ]
    key = parts[0]
    for p in parts[1:]:
        key = key + "\x1f" + p  # \x1f не встречается в тексте, склейка однозначна
    return key


def content_hash(df: pd.DataFrame) -> pd.Series:
    """Хэш контента объявления для поиска дублей.

    Одного заголовка мало (тысячи «Сантехник»), поэтому берём заголовок,
    описание, параметры, локацию и цену. md5 из stdlib: детерминирован
    между версиями pandas, в отличие от hash_pandas_object.
    """
    price = df["item_price"].map(lambda x: "" if pd.isna(x) else repr(float(x)))
    joined = (
        norm_text(df["item_title_raw"]) + "\x1f"
        + norm_text(df["item_description_raw"]) + "\x1f"
        + norm_text(df["item_infm_params_text"]) + "\x1f"
        + df["item_location_id"].astype(str) + "\x1f"
        + price
    )
    return joined.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:16])


def build_instances(train: pd.DataFrame) -> pd.DataFrame:
    """Строки train -> экземпляры запроса: одна строка на ключ QUERY_KEY.

    items = отсортированный список уникальных item_id: повторные выборы той же
    пары (6% строк train) в метрике не должны считаться дважды.
    qid = md5 ключа: стабилен между запусками и не зависит от порядка строк.
    """
    df = train.assign(qkey=query_key(train))
    g = df.groupby("qkey", sort=True)
    inst = g[QUERY_KEY].first()
    inst["items"] = g["item_id"].agg(lambda s: sorted(set(s)))
    inst = inst.reset_index()
    inst["qid"] = inst["qkey"].map(lambda k: hashlib.md5(k.encode()).hexdigest()[:16])
    assert inst["qid"].is_unique, "коллизия qid"
    inst["qtext"] = norm_text(inst["search_query"])
    return inst.drop(columns="qkey")


def query_text(q: pd.DataFrame) -> pd.Series:
    """Текст запроса для поиска: сам запрос + фильтры.

    Фильтры несут смысл (пустой запрос + «Вид услуги Красота»), поэтому склеиваем.
    """
    return (norm_text(q["search_query"]) + " " + norm_text(q["search_infm_params_text"])).str.strip()


def item_dense_text(c: pd.DataFrame, params_chars: int = 200, desc_chars: int = 300) -> pd.Series:
    """Текст объявления для эмбеддинга: заголовок первым, чтобы его не обрезало.

    Параметры длинные и шаблонные («Рабочие дни ...»), самое полезное
    («Вид услуги», «Тип услуги») обычно в начале, поэтому берём начало.
    """
    return (
        c["item_title_raw"].fillna("") + " | "
        + c["item_infm_params_text"].fillna("").str.slice(0, params_chars) + " | "
        + c["item_description_raw"].fillna("").str.slice(0, desc_chars)
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по сфере в км (векторно)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def location_centroids(corpus: pd.DataFrame) -> pd.DataFrame:
    """Центр локации = медиана координат объявлений корпуса в ней.

    Координат у запроса нет, только search_location_id; медиана устойчива к
    объявлениям с адресом в другом конце региона.
    """
    return corpus.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()


_REGION = re.compile(r"(област|обл\.|край|республик|автономн|район|р-н)", re.I)
_STREET = re.compile(r"^(ул\.|улица|пр\.|пр-т|проспект|пл\.|площадь|пер\.|переулок|ш\.|шоссе|наб\.|бул\.|б-р|мкр|д\.|\d)", re.I)
_PREFIX = re.compile(r"^(городской округ|муниципальное образование|город|г\.|посёлок|поселок|пгт|село|деревня)\s+", re.I)


def location_names(corpus: pd.DataFrame) -> pd.Series:
    """Название города для item_location_id: самый частый первый элемент адреса из
    «Место оказания услуг …», который не регион и не улица (иначе регион). Названий
    локаций в данных нет; это нужно только для текста cross-encoder («город: X»).
    Передавать объявления корпуса и train (только контент, без разметки): так
    покрываются локации запросов без объявлений в корпусе."""
    addr = corpus["item_infm_params_text"].fillna("").str.extract(r"Место оказания услуг (.+?)(?= [А-ЯЁ][а-яё]+ [а-яё]+ |$)")[0]

    def city(a):
        """Первый элемент адреса, не регион и не улица; если города нет — регион."""
        if not isinstance(a, str):
            return None
        parts = [_PREFIX.sub("", x.strip()).strip() for x in a.split(",")]
        cities = [x for x in parts if x and not _REGION.search(x) and not _STREET.search(x)]
        regions = [x for x in parts if x and _REGION.search(x)]
        return cities[0] if cities else (regions[0] if regions else None)

    c = pd.DataFrame({"loc": corpus["item_location_id"], "city": addr.map(city)}).dropna()
    names = c.groupby("loc")["city"].agg(lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0])
    return names


def query_location_names(item_names: pd.Series, history: pd.DataFrame, item_loc: pd.Series) -> pd.Series:
    """Название для search_location_id. Есть объявления в этой локации — их город.
    Нет (локация-регион, например 107620) — самый частый город объявлений, выбранных
    из неё в истории (для P2/P3 это P1 с прореживанием, разметка P2/P3 не используется)."""
    ex = history[["search_location_id", "items"]].explode("items")
    ex["city"] = ex["items"].map(item_loc).map(item_names)
    from_hist = ex.dropna().groupby("search_location_id")["city"].agg(
        lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0])
    return pd.concat([item_names, from_hist[~from_hist.index.isin(item_names.index)]])
