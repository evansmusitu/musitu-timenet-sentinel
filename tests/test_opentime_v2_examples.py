import pandas as pd

from sentinel.data import COLS
from sentinel import opentime_v2_examples as v2


def frame(units, length=120):
    return pd.DataFrame([{"unit": u, "cycle": t, **{c: float(u * 1000 + t + i) for i, c in enumerate(COLS[2:])}}
                         for u in units for t in range(1, length + 1)])


def test_full_history_is_past_only_and_uses_varying_sensors():
    dev = frame([1])
    rows = v2.training_rows(dev, seed=0)
    mutated = dev.copy()
    for r in rows:
        later = mutated.cycle > r["cycle"]
        changed = mutated.copy(); changed.loc[later, v2.SENSORS] = -1.0
        again = [x for x in v2.training_rows(changed, seed=0) if x["cycle"] == r["cycle"]][0]
        assert again["series"] == r["series"]
        assert len(r["series"]) == len(v2.SENSORS) == 14
        assert len(r["series"][0]["values"]) == r["cycle"]
        assert f"{r['cycle']} operating cycles" in r["prompt"]
        assert r["answer"] == ("Yes" if r["rul"] <= 30 else "No")


def test_training_rows_balanced_per_engine():
    rows = v2.training_rows(frame([1, 2], length=200), seed=0)
    yes = sum(r["answer"] == "Yes" for r in rows)
    assert yes == 2 * 11 and len(rows) - yes == 2 * 11


def test_official_rows_one_per_engine_with_nasa_label():
    test = frame([1, 2], length=40)
    out = v2.official_rows(test, {1: 12, 2: 55})
    assert [(r["unit"], r["cycle"], r["answer"]) for r in out] == [(1, 40, "Yes"), (2, 40, "No")]


def test_truncations_respect_minimum_history():
    out = v2.truncated_rows(frame([1, 2, 3], length=150), per_engine=10, seed=1)
    assert len(out) == 30 and min(r["cycle"] for r in out) >= v2.MIN_HISTORY


def test_threshold_and_confusion():
    t, f1 = v2.best_threshold([0.1, 0.9, 0.8, -1.0], [False, True, True, False])
    assert f1 == 1.0 and t == 0.8
    c = v2.confusion([True, False, True, False], [True, True, False, False])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
