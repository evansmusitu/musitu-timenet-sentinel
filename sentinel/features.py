"""Leakage-safe multivariate sensor window features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import COLS

SIGNALS = COLS[2:]

def window_features(w: pd.DataFrame, end_cycle: int) -> dict:
    result = {"age_cycles": float(end_cycle)}
    for col in SIGNALS:
        x = w[col].to_numpy(dtype=float)
        result[f"{col}_last"] = float(x[-1])
        result[f"{col}_mean"] = float(x.mean())
        result[f"{col}_std"] = float(x.std())
        result[f"{col}_slope"] = float((x[-1]-x[0]) / max(1, len(x)-1))
    return result

def build_examples(train: pd.DataFrame, test: pd.DataFrame | None = None, official_rul=None, *, window=30, stride=5) -> pd.DataFrame:
    if window < 2 or stride < 1:
        raise ValueError("Invalid window or stride")
    rows = []
    for source, frame in [("development", train), ("official_test", test)]:
        if frame is None:
            continue
        for unit, group in frame.groupby("unit", sort=True):
            group = group.sort_values("cycle").reset_index(drop=True)
            if len(group) < window:
                continue
            if source == "development":
                ends = range(window-1, len(group), stride)
                end_life = int(group.cycle.iloc[-1])
            else:
                if official_rul is None or int(unit) not in official_rul:
                    raise ValueError("Official RUL label missing")
                ends = [len(group)-1]
            for end in ends:
                cycle = int(group.cycle.iloc[end])
                observed = group.iloc[end-window+1:end+1]
                # Future engine lifetime is used ONLY to generate a label, never a feature.
                rul = end_life-cycle if source == "development" else int(official_rul[int(unit)])
                rows.append({"split_source":source, "unit":int(unit), "cycle":cycle,
                             "rul":float(rul), "failure_30":int(rul<=30),
                             **window_features(observed, cycle)})
    return pd.DataFrame(rows)

def split_engine_ids(df: pd.DataFrame, seed=42):
    ids = np.array(sorted(df.loc[df.split_source=="development", "unit"].unique()))
    if len(ids) < 2:
        raise ValueError("Need at least 2 engines")
    np.random.default_rng(seed).shuffle(ids)
    cutoff = max(1, min(len(ids)-1, int(0.8*len(ids))))
    return ids[:cutoff], ids[cutoff:]
