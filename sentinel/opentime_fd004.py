"""NASA C-MAPSS FD004 for the OpenTime v2 recipe (CPU-only data preparation, no model).

FD004 mixes six operating conditions and two fault modes. Raw sensor values jump whenever the
operating condition changes, which hides wear. Each sensor is therefore standardised within its
operating condition, using means and standard deviations from the development-TRAIN engines only.
The same statistics are applied unchanged to validation and official test engines.

The official test population matches the existing classical FD004 audit: engines with at least
30 observed cycles (237 of 248). Shorter engines are excluded and counted.
"""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .data import COLS, SOURCE_MD5
from .opentime_v2_examples import SENSORS

MIN_TEST_HISTORY = 30


def _table(z: ZipFile, name: str) -> pd.DataFrame:
    members = {Path(n).name: n for n in z.namelist()}
    df = pd.read_csv(io.BytesIO(z.read(members[name])), sep=r"\s+", header=None).dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise ValueError(f"{name}: expected 26 columns")
    df.columns = COLS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    if not np.isfinite(df.to_numpy(float)).all():
        raise ValueError(f"{name}: non-finite values")
    for _, g in df.groupby("unit"):
        if not np.array_equal(g.cycle.to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"{name}: non-contiguous cycles")
    return df


def condition(frame: pd.DataFrame) -> pd.Series:
    """Operating-condition key from the three settings (six clusters in FD002/FD004)."""
    return (frame.setting_1.round(0).astype(int).astype(str) + "_" +
            frame.setting_2.round(2).astype(str) + "_" + frame.setting_3.round(0).astype(int).astype(str))


def fit_condition_stats(train: pd.DataFrame) -> dict:
    c = condition(train)
    stats = {}
    for key, g in train.groupby(c):
        mean = g[SENSORS].mean(); std = g[SENSORS].std().replace(0, 1.0).fillna(1.0)
        stats[key] = (mean, std)
    if len(stats) != 6:
        raise ValueError(f"Expected 6 FD004 operating conditions, found {len(stats)}")
    return stats


def normalise(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = frame.copy()
    out[SENSORS] = out[SENSORS].astype(float)
    c = condition(frame)
    unknown = set(c.unique()) - set(stats)
    if unknown:
        raise ValueError(f"Unseen operating conditions: {sorted(unknown)}")
    for key, (mean, std) in stats.items():
        m = c == key
        out.loc[m, SENSORS] = (frame.loc[m, SENSORS] - mean) / std
    return out


def load_fd004(root: Path, partitions):
    """Return (dev_train, dev_val, test, labels, info), all condition-normalised with dev-train
    statistics. `partitions` is the same engine-disjoint 80/20 splitter used for FD001."""
    import hashlib
    archive = root / "CMAPSSData.zip"
    if hashlib.md5(archive.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError("Source MD5 mismatch")
    with ZipFile(archive) as z:
        train = _table(z, "train_FD004.txt"); test = _table(z, "test_FD004.txt")
        rul = pd.read_csv(io.BytesIO(z.read({Path(n).name: n for n in z.namelist()}["RUL_FD004.txt"])),
                          sep=r"\s+", header=None).dropna(axis=1, how="all")[0].astype(int).tolist()
    ids = sorted(test.unit.unique())
    if len(ids) != len(rul):
        raise ValueError("FD004 label count mismatch")
    labels = dict(zip(ids, rul))
    dev_a, dev_b = partitions(train)
    stats = fit_condition_stats(dev_a)
    lengths = test.groupby("unit").size()
    eligible = set(int(u) for u in lengths[lengths >= MIN_TEST_HISTORY].index)
    test = test[test.unit.isin(eligible)]
    info = {"train_engines": int(dev_a.unit.nunique()), "val_engines": int(dev_b.unit.nunique()),
            "official_test_engines": len(ids), "eligible_test_engines": len(eligible),
            "excluded_short_history": len(ids) - len(eligible),
            "normalisation": "per operating condition, development-train statistics only"}
    return (normalise(dev_a, stats), normalise(dev_b, stats), normalise(test, stats),
            {u: labels[u] for u in sorted(eligible)}, info)
