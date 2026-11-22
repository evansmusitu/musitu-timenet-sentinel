"""Reproducible, paired evaluation of *independent* FD001 CPU model artifacts.

This module never trains on, tunes to, or re-scores labels from the held-out
NASA 100-engine test set. It only audits already frozen model predictions.

The public JSON is aggregate-only: do not republish NASA source rows or 100
individual labels until source redistribution terms have been reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_BASELINE = {"unit", "cycle", "rul", "predicted_rul", "warning_30"}
EXPECTED_TEMPORAL = {
    "unit", "observed_cycle", "actual_rul", "predicted_rul",
    "risk_within_30_probability", "risk_within_30_predicted",
}
SEED = 20261122
THRESHOLD = 30
MODEL_LINKS = {
    "classical": "https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482",
    "compact_temporal": "https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    for name in columns:
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{source}: {name} contains a missing or non-finite value")


def _boolean(frame: pd.DataFrame, column: str, source: str) -> np.ndarray:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.lower().str.strip()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError(f"{source}: {column} requires explicit true/false labels")
    return normalized.eq("true").to_numpy(dtype=bool)


def load_paired_artifacts(baseline: Path, temporal: Path, expected_engines: int = 100) -> pd.DataFrame:
    """Join by engine identifier and reject a partial, mismatched or altered pair."""
    base = pd.read_csv(baseline)
    temp = pd.read_csv(temporal)
    for frame, required, source in [
        (base, EXPECTED_BASELINE, "baseline"),
        (temp, EXPECTED_TEMPORAL, "temporal"),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source}: missing required columns {sorted(missing)}")
        if frame.empty or frame.unit.isna().any() or frame.unit.duplicated().any():
            raise ValueError(f"{source}: engine IDs must be unique and nonempty")
        _finite_numeric(frame, ("unit",), source)
        if not np.array_equal(frame.unit, frame.unit.astype(int)) or (frame.unit <= 0).any():
            raise ValueError(f"{source}: engine IDs must be positive integers")
    _finite_numeric(base, ("cycle", "rul", "predicted_rul"), "baseline")
    _finite_numeric(temp, ("observed_cycle", "actual_rul", "predicted_rul", "risk_within_30_probability"), "temporal")
    if len(base) != expected_engines or len(temp) != expected_engines:
        raise ValueError(f"Expected exactly {expected_engines} engine endpoints")
    if (base[["cycle", "rul", "predicted_rul"]] < 0).any().any() or (
        temp[["observed_cycle", "actual_rul", "predicted_rul"]] < 0
    ).any().any():
        raise ValueError("Cycles, ground-truth RUL, and predicted RUL cannot be negative")
    if not (0 <= temp.risk_within_30_probability).all() or not (
        temp.risk_within_30_probability <= 1
    ).all():
        raise ValueError("Temporal risk probabilities must lie in [0,1]")
    base["warning_30"] = _boolean(base, "warning_30", "baseline")
    temp["risk_within_30_predicted"] = _boolean(temp, "risk_within_30_predicted", "temporal")
    if not np.array_equal(base.warning_30.to_numpy(dtype=bool), (base.predicted_rul <= THRESHOLD).to_numpy()):
        raise ValueError("Baseline warning flag disagrees with the frozen 30-cycle threshold")
    if not np.array_equal(temp.risk_within_30_predicted.to_numpy(dtype=bool), (temp.risk_within_30_probability >= 0.5).to_numpy()):
        raise ValueError("Temporal warning flag disagrees with the frozen 0.5 decision threshold")
    merged = base.merge(temp, on="unit", how="inner", validate="one_to_one", suffixes=("_base", "_temporal"))
    if len(merged) != expected_engines:
        raise ValueError("Model artifact engine ID sets disagree")
    if not np.array_equal(merged.cycle.to_numpy(), merged.observed_cycle.to_numpy()):
        raise ValueError("Model artifact observation cycles disagree")
    if not np.array_equal(merged.rul.to_numpy(), merged.actual_rul.to_numpy()):
        raise ValueError("Model artifact official test labels disagree")
    return merged.sort_values("unit").reset_index(drop=True)


def _counts(y: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum(y & predicted)),
        "fp": int(np.sum(~y & predicted)),
        "fn": int(np.sum(y & ~predicted)),
        "tn": int(np.sum(~y & ~predicted)),
    }


def _scores(count: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _bootstrap_paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, abs_a: np.ndarray, abs_b: np.ndarray,
                      *, samples: int = 10000, seed: int = SEED) -> dict:
    """Engine-level paired bootstrap; no re-training or threshold selection."""
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    count = len(y)
    gen = np.random.default_rng(seed)
    deltas_f1 = np.empty(samples, dtype=float)
    deltas_mae = np.empty(samples, dtype=float)
    # Small batches bound memory if the evaluator later handles larger datasets.
    for start in range(0, samples, 1024):
        end = min(start + 1024, samples)
        ix = gen.integers(0, count, size=(end-start, count))
        sampled_y = y[ix]
        def sampled_f1(pred: np.ndarray) -> np.ndarray:
            pp = pred[ix]
            tp = (sampled_y & pp).sum(axis=1)
            fp = ((~sampled_y) & pp).sum(axis=1)
            fn = (sampled_y & ~pp).sum(axis=1)
            denom = 2 * tp + fp + fn
            return np.divide(2 * tp, denom, out=np.zeros(len(ix), dtype=float), where=denom > 0)
        deltas_f1[start:end] = sampled_f1(b) - sampled_f1(a)
        deltas_mae[start:end] = (abs_a[ix] - abs_b[ix]).mean(axis=1)
    def interval(x: np.ndarray) -> dict[str, float]:
        lo, hi = np.quantile(x, [.025, .975])
        return {"lower": float(lo), "upper": float(hi)}
    return {
        "method": "paired engine-level percentile bootstrap; 95% interval",
        "seed": seed, "resamples": samples,
        "temporal_minus_baseline_warning_f1": interval(deltas_f1),
        "baseline_minus_temporal_mae_cycles": interval(deltas_mae),
        "important": "Intervals include uncertainty from the 100 test engines only; they do not measure dataset shift, deployment safety or model training variance.",
    }


def evaluate(paired: pd.DataFrame, *, samples: int = 10000, seed: int = SEED) -> dict:
    """Evaluate two *frozen* predictors using exactly one endpoint per engine."""
    y = (paired.rul <= THRESHOLD).to_numpy(dtype=bool)
    a = paired.warning_30.to_numpy(dtype=bool)
    b = paired.risk_within_30_predicted.to_numpy(dtype=bool)
    true_rul = paired.rul.to_numpy(dtype=float)
    abs_a = np.abs(paired.predicted_rul_base.to_numpy(dtype=float) - true_rul)
    abs_b = np.abs(paired.predicted_rul_temporal.to_numpy(dtype=float) - true_rul)
    def result(predicted: np.ndarray, errors: np.ndarray) -> dict:
        return {
            "confusion": _counts(y, predicted),
            "warning_30": _scores(_counts(y, predicted)),
            "rul_mae_cycles": float(errors.mean()),
            "rul_rmse_cycles": float(np.sqrt(np.mean(errors ** 2))),
        }
    base = result(a, abs_a)
    temporal = result(b, abs_b)
    temporal["risk_probability_brier_score"] = float(np.mean((paired.risk_within_30_probability.to_numpy(dtype=float) - y.astype(float)) ** 2))
    ci = _bootstrap_paired(y, a, b, abs_a, abs_b, samples=samples, seed=seed)
    return {
        "project": "MUSITU AXIOM Sentinel",
        "dataset": "NASA C-MAPSS FD001 simulated engines",
        "test_population": {"independent_engine_endpoints": int(len(y)), "rul_at_most_30": int(y.sum()), "rul_above_30": int((~y).sum())},
        "fixed_operating_point": {"warning_horizon_cycles": THRESHOLD, "compact_warning_head_probability_threshold": 0.5,
                                  "test_labels_used_for_threshold_selection": False},
        "baseline": base,
        "compact_temporal_qa": temporal,
        "paired_observed_difference": {
            "additional_true_warnings": temporal["confusion"]["tp"] - base["confusion"]["tp"],
            "additional_false_alarms": temporal["confusion"]["fp"] - base["confusion"]["fp"],
            "fewer_missed_warnings": base["confusion"]["fn"] - temporal["confusion"]["fn"],
            "warning_f1_temporal_minus_baseline": temporal["warning_30"]["f1"] - base["warning_30"]["f1"],
            "rul_mae_baseline_minus_temporal_cycles": base["rul_mae_cycles"] - temporal["rul_mae_cycles"],
        },
        "bootstrap_uncertainty": ci,
        "illustrative_penalty_scenarios": [
            {"miss_cost_units_per_engine": ratio, "false_alarm_cost_units_per_engine": 1,
             "baseline_penalty_units": base["confusion"]["fn"] * ratio + base["confusion"]["fp"],
             "compact_penalty_units": temporal["confusion"]["fn"] * ratio + temporal["confusion"]["fp"]}
            for ratio in (1, 2, 5, 10, 20)
        ],
        "provenance": {
            "baseline_workflow": MODEL_LINKS["classical"],
            "compact_temporal_workflow": MODEL_LINKS["compact_temporal"],
            "baseline_prediction_file": "official_test_predictions.csv",
            "compact_prediction_file": "temporal_qa_predictions.csv",
            "both_artifacts_matched_by": ["unit", "observed_cycle", "official_test_rul"],
        },
        "limitations": [
            "Both evaluated models are independent CPU-trained systems, NOT the organizer's OpenTime 5B model.",
            "This is a simulated NASA FD001 benchmark, not real industrial validation or a cross-dataset test.",
            "The benchmark contains one final endpoint per test engine, not continuous streaming warning lead times.",
            "The penalty ratios are illustrative dimensionless choices, not measured downtime, costs or savings.",
            "The paired bootstrap interval is descriptive; crossing zero means the apparent advantage is uncertain.",
            "No underlying NASA source data, per-engine labels or model weights are included in this aggregate report.",
        ],
    }


def run(baseline: Path, temporal: Path, output: Path, *, samples: int = 10000, expected_engines: int = 100) -> dict:
    paired = load_paired_artifacts(baseline, temporal, expected_engines=expected_engines)
    report = evaluate(paired, samples=samples)
    report["provenance"]["baseline_predictions_sha256"] = _digest(baseline)
    report["provenance"]["compact_predictions_sha256"] = _digest(temporal)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path, help="Exact baseline run predictions CSV")
    p.add_argument("--temporal", required=True, type=Path, help="Exact independent compact QA model predictions CSV")
    p.add_argument("--output", required=True, type=Path, help="Aggregate-only audit JSON destination")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--expect-engines", type=int, default=100)
    args = p.parse_args()
    print(json.dumps(run(args.baseline, args.temporal, args.output, samples=args.bootstrap_samples,
                         expected_engines=args.expect_engines), indent=2))


if __name__ == "__main__":
    main()
