import pytest

from src.validate_answer import validate

Q = {"Aa01", "bB02"}  # смешанный регистр: регистр менять нельзя
C = {f"{i:016x}" for i in range(200)}
IDS = sorted(C)


def _write(tmp_path, text):
    p = tmp_path / "answer.csv"
    p.write_text(text, encoding="utf-8")
    return p


def _ok():
    return "query_id,answer\n" + "".join(f"{q},{' '.join(IDS[i * 50:(i + 1) * 50])}\n" for i, q in enumerate(sorted(Q)))


def test_valid(tmp_path):
    validate(_write(tmp_path, _ok()), Q, C)


@pytest.mark.parametrize("broken", [
    lambda s: s.replace("query_id,answer", ",query_id,answer"),          # индекс
    lambda s: s.replace("Aa01", "aa01"),                                 # регистр id
    lambda s: s.replace(IDS[0] + " ", IDS[0] + "  ", 1),                 # двойной пробел
    lambda s: s.replace(IDS[1], IDS[0], 1),                              # повтор id
    lambda s: s.replace(IDS[0], "ffffffffffffffff", 1),                  # id не из корпуса
    lambda s: s.replace(" " + IDS[49], "", 1),                           # 49 id
    lambda s: s + s.splitlines()[1] + "\n",                              # лишняя строка
])
def test_broken(tmp_path, broken):
    with pytest.raises(AssertionError):
        validate(_write(tmp_path, broken(_ok())), Q, C)
