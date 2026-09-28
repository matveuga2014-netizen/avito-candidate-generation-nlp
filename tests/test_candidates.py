import pandas as pd

from src.candidates import rrf


def test_rrf_weights_fill_and_tiebreak():
    rows = [("q", "a", 1, "x"), ("q", "b", 2, "x"), ("q", "b", 1, "y"), ("q", "c", 1, "z")]
    # pop_loc — фолбэк на 50 объявлений
    rows += [("q", f"p{i:02d}", i + 1, "pop_loc") for i in range(50)]
    c = pd.DataFrame(rows, columns=["qid", "item_id", "rank", "channel"])
    out = rrf(c, {"x": 1.0, "y": 1.0, "z": 0.0})
    assert len(out["q"]) == 50 == len(set(out["q"]))
    assert out["q"][:2] == ["b", "a"]          # b набрал из двух каналов
    assert "c" not in out["q"]                 # канал с весом 0 не участвует
    assert out["q"][2:4] == ["p00", "p01"]     # добивка в порядке pop_loc
    # равные score -> порядок по item_id
    c2 = pd.DataFrame([("q", "k", 1, "x"), ("q", "j", 1, "y")] + rows[4:], columns=c.columns)
    assert rrf(c2, {"x": 1.0, "y": 1.0})["q"][:2] == ["j", "k"]
