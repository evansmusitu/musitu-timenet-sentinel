"""Full-history OpenTime examples for the 30-cycle warning question (CPU-only, no model).

Changes from the pilot (`opentime_examples.py`):
- the model sees each engine's whole past history, not only the last 30 cycles, because the
  checkpoint rescales every series to mean 0 / std 1 and a 30-cycle window hides slow wear;
- only the 14 FD001 sensors that vary are sent; the 7 constant sensors and 3 settings are dropped;
- the prompt states the engine's current cycle count, which is known at prediction time;
- training uses every development engine, with more positives and near-threshold negatives.

Everything here is past-only: a window that ends at cycle t uses rows 1..t of that engine only.
"""
from __future__ import annotations

import random

import numpy as np

# FD001 sensors with non-zero variance on the development engines (checked in tests and at run
# time). Sensors 1, 5, 6, 10, 16, 18, 19 and the three operating settings are constant or noise.
SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
HORIZON = 30
MIN_HISTORY = 31  # the official test histories are at least 31 cycles long


def question(cycle: int) -> str:
    return (
        f"This simulated turbofan engine has completed {cycle} operating cycles so far. "
        f"Will it reach end of useful life within the next {HORIZON} operating cycles? "
        "Answer Yes or No."
    )


def row(group, end: int, rul: int, unit: int) -> dict:
    """One example from rows 0..end (inclusive) of one engine's sorted history."""
    history = group.iloc[: end + 1]
    cycle = int(history.cycle.iloc[-1])
    return {
        "unit": int(unit),
        "cycle": cycle,
        "rul": int(rul),
        "answer": "Yes" if rul <= HORIZON else "No",
        "prompt": question(cycle),
        "series": [{"name": c, "values": history[c].astype(float).tolist()} for c in SENSORS],
    }


def _engines(frame):
    for unit, group in frame.groupby("unit", sort=True):
        yield int(unit), group.sort_values("cycle").reset_index(drop=True)


def training_rows(frame, seed: int = 42, pos_step: int = 3, near: int = 6, far: int = 5) -> list[dict]:
    """Per run-to-failure engine: every `pos_step`-th window with RUL <= 30, `near` windows with
    RUL 31..90 and `far` random windows with RUL > 90. Shuffled, deterministic."""
    rng = random.Random(seed)
    out = []
    for unit, g in _engines(frame):
        last = len(g) - 1
        ends = [e for e in range(MIN_HISTORY - 1, last + 1)]
        rul = {e: last - e for e in ends}
        pos = [e for e in ends if rul[e] <= HORIZON][::pos_step]
        mid = [e for e in ends if HORIZON < rul[e] <= 90]
        hi = [e for e in ends if rul[e] > 90]
        mid = mid[:: max(1, len(mid) // near)][:near] if mid else []
        hi = rng.sample(hi, min(far, len(hi)))
        out += [row(g, e, rul[e], unit) for e in pos + mid + hi]
    rng.shuffle(out)
    return out


def truncated_rows(frame, per_engine: int, seed: int) -> list[dict]:
    """Mimic the official test protocol on development engines: cut each run-to-failure history
    at `per_engine` random points (history >= 31 cycles) and ask about the cut point."""
    rng = random.Random(seed)
    out = []
    for unit, g in _engines(frame):
        last = len(g) - 1
        ends = rng.sample(range(MIN_HISTORY - 1, last + 1), min(per_engine, last - MIN_HISTORY + 2))
        out += [row(g, e, last - e, unit) for e in sorted(ends)]
    return out


def official_rows(test, official_rul: dict) -> list[dict]:
    """One example per official test engine: its full provided history and NASA's RUL label."""
    out = [row(g, len(g) - 1, int(official_rul[unit]), unit) for unit, g in _engines(test)]
    if len(out) != len(official_rul):
        raise ValueError("Official test engines and labels do not match")
    return out


def check_sensors(frame) -> None:
    std = frame[SENSORS].std()
    if (std <= 1e-6).any():
        raise ValueError(f"Constant sensors selected: {list(std[std <= 1e-6].index)}")


def best_threshold(scores, actual) -> tuple[float, float]:
    """Threshold on the Yes-vs-No score that maximises F1 (ties: the higher threshold)."""
    scores, actual = np.asarray(scores, float), np.asarray(actual, bool)
    best = (-np.inf, 0.0)
    for t in np.unique(scores):
        pred = scores >= t
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1] or (f1 == best[1] and t > best[0]):
            best = (float(t), f1)
    return best


def confusion(pred, actual) -> dict:
    pred, actual = np.asarray(pred, bool), np.asarray(actual, bool)
    tp = int((pred & actual).sum()); fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum()); tn = int((~pred & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": (tp + tn) / len(actual)}
