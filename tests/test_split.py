import pandas as pd

from src.split import assign_parts


def _inst():
    # 300 текстов, у части по нескольку экземпляров
    rows = []
    for t in range(300):
        for j in range(1 + t % 4):
            rows.append({"qid": f"{t:04d}{j:012d}", "qtext": f"text {t}"})
    return pd.DataFrame(rows)


def test_parts_invariants():
    inst = _inst()
    part = assign_parts(inst, n_p2=40, n_p3=30)
    p2, p3 = inst[part == "P2"], inst[part == "P3"]
    assert len(p2) == 40 and len(p3) == 30
    assert p2["qtext"].is_unique and p3["qtext"].is_unique
    assert not set(p2["qtext"]) & set(p3["qtext"])
    # Другие экземпляры текстов P2/P3 остаются в истории: это «известные» запросы
    known = p3["qtext"].isin(inst.loc[part == "P1", "qtext"]).mean()
    assert 0 < known < 1


def test_deterministic_and_order_independent():
    inst = _inst()
    a = assign_parts(inst, 40, 30)
    shuffled = inst.sample(frac=1, random_state=1)
    b = assign_parts(shuffled, 40, 30)
    assert a.to_dict() == b.reindex(a.index).to_dict()


def test_thin_mask_only_p1_and_deterministic():
    from src.split import thin_mask
    sp = pd.DataFrame({
        "part": ["P1", "P1", "P1", "P2", "P3"],
        "items": [["a"], ["b"], ["z"], ["a"], ["b"]],
    })
    all_ = thin_mask(sp, share=1.0)
    assert all_.tolist() == [True, True, False, False, False]  # z не релевантно в P2/P3
    assert not thin_mask(sp, share=0.0).any()
    assert thin_mask(sp, 0.5, seed=7).equals(thin_mask(sp, 0.5, seed=7))
