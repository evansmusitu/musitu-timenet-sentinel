import pandas as pd
import pytest

from sentinel.data import COLS
from sentinel import opentime_fd004 as f4
from sentinel.opentime_v2_examples import SENSORS

CONDS = [(0, 0.0, 100), (10, 0.25, 100), (20, 0.7, 100), (25, 0.62, 60), (35, 0.84, 100), (42, 0.84, 100)]


def frame(units, length=12, offset=0.0):
    rows = []
    for u in units:
        for t in range(1, length + 1):
            s1, s2, s3 = CONDS[t % 6]
            rows.append({"unit": u, "cycle": t, "setting_1": s1, "setting_2": s2, "setting_3": s3,
                         **{c: float(100 * s1 + t + offset) for c in COLS[5:]}})
    return pd.DataFrame(rows)


def test_condition_stats_come_from_training_only():
    train = frame([1, 2]); test = frame([3], offset=1000.0)
    stats = f4.fit_condition_stats(train)
    a = f4.normalise(train, stats); b = f4.normalise(test, stats)
    assert abs(a[SENSORS].to_numpy().mean()) < 1e-9
    assert b[SENSORS].to_numpy().min() > 100  # test shift is kept, not re-centred


def test_unseen_condition_rejected():
    stats = f4.fit_condition_stats(frame([1]))
    odd = frame([2]); odd.loc[0, "setting_1"] = 99
    with pytest.raises(ValueError):
        f4.normalise(odd, stats)
