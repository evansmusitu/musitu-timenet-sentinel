"""Aggregate-only paired audit: OpenTime 5B v2 fine-tune vs the classical baseline (CPU only).

Inputs are the exact per-engine artifacts: the v2 job's `test_predictions.jsonl` (recovered from
the Nebius job log) and the baseline's `official_test_predictions.csv`. The output holds counts,
metrics and paired bootstrap intervals only; no per-engine rows or NASA labels are written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .opentime_v2_examples import confusion

SEED = 20261122


def _f1(y, p):
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def run(v2_predictions: Path, v2_metrics: Path, baseline: Path, output: Path, samples: int = 10000) -> dict:
    rows = pd.read_json(v2_predictions, lines=True).sort_values("unit")
    metrics = json.loads(v2_metrics.read_text())
    base = pd.read_csv(baseline).sort_values("unit")
    if list(rows.unit) != list(base.unit) or len(rows) != 100:
        raise ValueError("Prediction files do not cover the same 100 official engines")
    if not (rows.rul.to_numpy() == base.rul.to_numpy()).all():
        raise ValueError("RUL labels differ between artifacts")
    y = rows.actual.eq("Yes").to_numpy()
    threshold = float(metrics["threshold_from_validation"])
    tslm = rows.score.to_numpy() >= threshold
    tslm_gen = rows.generated.eq("Yes").to_numpy()
    gb = base.warning_30.astype(bool).to_numpy()
    rng = np.random.default_rng(SEED)
    diffs, diffs_gen = [], []
    for _ in range(samples):
        i = rng.integers(0, len(y), len(y))
        diffs.append(_f1(y[i], tslm[i]) - _f1(y[i], gb[i]))
        diffs_gen.append(_f1(y[i], tslm_gen[i]) - _f1(y[i], gb[i]))
    ci = lambda d: {"lower": float(np.percentile(d, 2.5)), "upper": float(np.percentile(d, 97.5))}
    report = {
        "dataset": "NASA C-MAPSS FD001 simulated engines, 100 official test engines (25 within 30 cycles)",
        "opentime_v2": {
            "checkpoint": metrics["checkpoint"], "revision": metrics["revision"],
            "train_examples": metrics["train_examples"], "train_engines": metrics["train_engines"],
            "epochs": metrics["epochs"], "gpu": metrics["gpu"],
            "threshold_from_validation": threshold,
            "validation_threshold": confusion(tslm, y),
            "greedy_generation": confusion(tslm_gen, y),
            "invalid_outputs": int(rows.generated.isna().sum()),
            "test_auroc": metrics["test"]["auroc"],
        },
        "classical_baseline": confusion(gb, y),
        "paired_bootstrap_95": {
            "opentime_threshold_minus_baseline_f1": ci(diffs),
            "opentime_generation_minus_baseline_f1": ci(diffs_gen),
            "resamples": samples, "seed": SEED,
            "note": "Engine-level resampling of the 100 test engines only; excludes training variance and dataset shift.",
        },
        "provenance": {
            "v2_predictions_sha256": hashlib.sha256(v2_predictions.read_bytes()).hexdigest(),
            "baseline_predictions_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "test_labels_used_for_training_or_threshold": False,
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-predictions", type=Path, required=True)
    p.add_argument("--v2-metrics", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.v2_predictions, a.v2_metrics, a.baseline, a.output), indent=2))


if __name__ == "__main__":
    main()
