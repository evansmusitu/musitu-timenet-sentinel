"""Frozen, longitudinal FD001 maintenance warning evaluation (CPU only).

Measures alerts throughout OBSERVED history, not just each official test
engine's last endpoint. No retraining, test-threshold tuning or OpenTime 5B
claims. Raw NASA telemetry is never published with the resulting summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLS, download_dataset, load_fd001
from .temporal_qa import predict

WINDOW = 30
STRIDE = 5
HORIZON = 30
PROBABILITY_THRESHOLD = 0.5


def make_history(test: pd.DataFrame, offsets: dict[int, int], *, stride: int = STRIDE):
    """Create past-only windows, including every test engine's final endpoint.

    At an earlier observation, actual RUL is the official last-observation
    offset PLUS the number of observed cycles remaining until that endpoint.
    Neither the future sensor measurements nor labels are model input.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    xs, rows = [], []
    for unit, group in test.groupby("unit", sort=True):
        unit = int(unit)
        if unit not in offsets:
            raise ValueError(f"missing official RUL label for engine {unit}")
        offset = offsets[unit]
        if isinstance(offset, (float, np.floating)) and not float(offset).is_integer():
            raise ValueError("official RUL labels must be integral nonnegative values")
        if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or int(offset) < 0:
            raise ValueError("official RUL labels must be nonnegative integers")
        if len(group) < WINDOW:
            raise ValueError(f"short engine history: {unit}")
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(f"non-contiguous engine history: {unit}")
        values = group[COLS[2:]].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite source telemetry")
        ends = list(range(WINDOW - 1, len(group), stride))
        if ends[-1] != len(group) - 1:
            ends.append(len(group) - 1)
        for end in ends:
            current = end + 1
            xs.append(values[end + 1 - WINDOW:end + 1].copy())
            rows.append((unit, current, int(offset) + len(group) - current))
    if not xs:
        raise ValueError("No engine windows produced")
    return np.stack(xs), pd.DataFrame(rows, columns=["unit", "cycle", "actual_rul"])


def summarize_replay(frame: pd.DataFrame, *, expected_engines: int = 100) -> dict:
    """Correlated-window and independent-engine counts at the fixed threshold."""
    needed = {"unit", "cycle", "actual_rul", "predicted_rul", "warning_probability"}
    if needed.difference(frame.columns) or frame.empty:
        raise ValueError("Replay missing required fields or empty")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle replay rows")
    numeric = frame[list(needed)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite replay fields")
    if (frame.unit <= 0).any() or (frame.cycle < WINDOW).any() or (frame.actual_rul < 0).any():
        raise ValueError("Invalid engine, cycle or RUL labels")
    prob = frame.warning_probability.to_numpy(dtype=float)
    if (prob < 0).any() or (prob > 1).any():
        raise ValueError("Warning probability outside 0..1")
    if frame.unit.nunique() != expected_engines:
        raise ValueError("Incomplete engine set for replay")
    order = frame.sort_values(["unit", "cycle"])
    truth = order.actual_rul.to_numpy() <= HORIZON
    warning = order.warning_probability.to_numpy() >= PROBABILITY_THRESHOLD
    confusion = {"tp": int((truth & warning).sum()),
                 "fp": int((~truth & warning).sum()),
                 "fn": int((truth & ~warning).sum()),
                 "tn": int((~truth & ~warning).sum())}
    eligible, detected, early_false, no_observed = 0, 0, 0, 0
    for unit, group in order.groupby("unit", sort=True):
        inside = group.actual_rul <= HORIZON
        pred = group.warning_probability >= PROBABILITY_THRESHOLD
        if inside.any():
            eligible += 1
            if (inside & pred).any():
                detected += 1
        else:
            no_observed += 1
        if ((~inside) & pred).any():
            early_false += 1
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    return {
        "observed_windows": int(len(order)),
        "window_confusion": confusion,
        "window_warning_f1": float((2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0),
        "engines": {
            "total": int(expected_engines),
            "with_in_horizon_observations": eligible,
            "with_in_horizon_warning": detected,
            "with_in_horizon_no_warning": eligible-detected,
            "with_out_of_horizon_false_alert": early_false,
            "without_observed_in_horizon_data": no_observed,
        },
    }


def make_cases(frame: pd.DataFrame, *, max_cases: int = 8) -> list[dict]:
    """First two engine IDs per outcome category, selected reproducibly."""
    labels = ("caught_risk", "missed_risk", "premature_alert", "clean_monitoring")
    groups = {label: [] for label in labels}
    for unit, group in frame.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        inside = group.actual_rul <= HORIZON
        flag = group.warning_probability >= PROBABILITY_THRESHOLD
        if (inside & flag).any():
            category = "caught_risk"
        elif inside.any():
            category = "missed_risk"
        elif ((~inside) & flag).any():
            category = "premature_alert"
        else:
            category = "clean_monitoring"
        groups[category].append({
            "engine_id": int(unit), "category": category,
            "any_premature_alert": bool(((~inside) & flag).any()),
            "sampled_history": [
                {"cycle": int(row.cycle),
                 "warning_probability": round(float(row.warning_probability), 5),
                 "predicted_rul": round(float(row.predicted_rul), 2),
                 "actual_rul": int(row.actual_rul)}
                for row in group.itertuples(index=False)
            ],
        })
    result=[]
    for k in range(2):
        for label in labels:
            if k<len(groups[label]) and len(result)<max_cases:
                result.append(groups[label][k])
    return result


def run(data_dir: Path, artifact_dir: Path, output_dir: Path, *,
        download=False, stride=STRIDE) -> dict:
    import torch
    from .sequence_integrity import (_load_compact, _verified_file, CHECKPOINT_SHA256,
                                     TEMPORAL_SHA256, verify_frozen_reproduction)
    from .temporal_qa import windows
    torch.set_num_threads(min(2,torch.get_num_threads()))
    checkpoint = artifact_dir / "temporal_qa_checkpoint.pt"
    reference_path = artifact_dir / "temporal_qa_predictions.csv"
    _verified_file(checkpoint,CHECKPOINT_SHA256)
    _verified_file(reference_path,TEMPORAL_SHA256)
    if download:
        download_dataset(data_dir)
    _, test, offsets = load_fd001(data_dir)
    if test.unit.nunique() != 100:
        raise ValueError("Expected exactly 100 held-out NASA engines")
    model, normalization = _load_compact(checkpoint)
    endpoint_x, endpoint_rul, endpoint_meta = windows(
        test,set(test.unit.unique()),official_rul=offsets)
    endpoint_predictions, endpoint_warnings, endpoint_prob = predict(
        model,endpoint_x,normalization)
    verify_frozen_reproduction(pd.read_csv(reference_path),endpoint_meta,endpoint_rul,
                               endpoint_predictions,endpoint_warnings,endpoint_prob)
    sequences, frame = make_history(test, offsets, stride=stride)
    model_rul, model_warning, model_prob = predict(model,sequences,normalization)
    frame["predicted_rul"] = model_rul
    frame["warning_probability"] = model_prob
    final = frame.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
    if not np.array_equal(final[["unit","cycle"]].to_numpy(),np.asarray(endpoint_meta)):
        raise ValueError("Replay does not include all original final endpoints")
    if not np.allclose(final.predicted_rul,endpoint_predictions,rtol=0,atol=1e-4):
        raise ValueError("Replay differs from frozen endpoint predictions")
    if not np.allclose(final.warning_probability,endpoint_prob,rtol=0,atol=1e-5):
        raise ValueError("Replay differs from frozen endpoint warning probabilities")
    result = {
        "project": "MUSITU AXIOM Sentinel",
        "study": "Post-hoc frozen compact-model history replay",
        "dataset": "NASA C-MAPSS FD001, 100 held-out simulated engines",
        "model": "Independent compact temporal QA Transformer; NOT OpenTime 5B",
        "sample_stride_cycles": stride,
        "window_length_cycles": WINDOW,
        "warning_horizon_cycles": HORIZON,
        "fixed_probability_threshold": PROBABILITY_THRESHOLD,
        "frozen_checkpoint_sha256": CHECKPOINT_SHA256,
        "frozen_prediction_sha256": TEMPORAL_SHA256,
        "original_endpoint_predictions_reproduced": True,
        "heldout_frozen_model_no_retraining": True,
        "metrics": summarize_replay(frame),
        "cases": make_cases(frame),
        "limitations": [
            "Multiple windows per engine are correlated; pooled-window metrics are not independent-engine confidence estimates.",
            "Official FD001 test trajectories stop before failure; many engines have no observed data within 30 cycles of failure.",
            "A past-only rolling replay is not a deployed live monitor and has no measured operational benefit.",
            "Some early warnings are premature false alerts under the fixed 30-cycle horizon.",
            "Only selected prediction traces are included in the public report; no raw source sensor rows are redistributed.",
            "This uses the independent compact model, NOT the organizer's OpenTime 5B pilot.",
            "Exploratory analysis uses the fixed 0.5 threshold; no test-threshold tuning.",
        ],
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"longitudinal_replay.json").write_text(json.dumps(result,indent=2)+"\n")
    frame.to_csv(output_dir/"longitudinal_replay_full.csv",index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    parser.add_argument("--artifact-dir",type=Path,default=Path("artifacts/frozen_compact"))
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/longitudinal"))
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--stride",type=int,default=STRIDE)
    args=parser.parse_args()
    print(json.dumps(run(args.data_dir,args.artifact_dir,args.output_dir,download=args.download,stride=args.stride)["metrics"],indent=2))


if __name__=="__main__":
    main()
