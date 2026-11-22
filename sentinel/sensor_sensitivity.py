"""Noncausal channel-mean replacement sensitivity for the original frozen compact FD001 model.

Not fault diagnosis, causal explanation, or OpenTime 5B inference. Every reported
model score must reproduce its original frozen benchmark before interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import COLS, download_dataset, load_fd001
from .temporal_qa import predict, windows

CHANNELS = COLS[2:]
SELECTED_CATEGORY = ("caught_risk", "missed_risk", "premature_alert", "clean_monitoring")


def ablate_channels(x, training_mean, baseline_probability, risk_predict_fn):
    """Return [24, N] risk after replacing one complete input channel with its training mean."""
    arr = np.asarray(x, dtype=np.float32)
    mean = np.asarray(training_mean, dtype=np.float32)
    original = np.asarray(baseline_probability, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 2 or arr.shape[2] != 24:
        raise ValueError("Expected [N, history>=2, 24 channels] data")
    if mean.shape != (1, 1, 24):
        raise ValueError("training mean shape must be [1,1,24]")
    if original.shape != (len(arr),):
        raise ValueError("baseline probability length does not match input windows")
    if not np.isfinite(arr).all() or not np.isfinite(mean).all() or not np.isfinite(original).all():
        raise ValueError("Inputs and baseline must be finite")
    if (original < 0).any() or (original > 1).any():
        raise ValueError("Invalid baseline probability")
    results = np.empty((24, len(arr)), dtype=float)
    for channel in range(24):
        perturbed = arr.copy()
        perturbed[:, :, channel] = float(mean[0, 0, channel])
        predicted = np.asarray(risk_predict_fn(perturbed), dtype=float)
        if predicted.shape != (len(arr),):
            raise ValueError("Perturbed prediction count differs from input")
        if not np.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
            raise ValueError("Perturbed probability outside 0..1 or not finite")
        results[channel] = predicted
    return results


def summarized_channels(ablated_probability, original_probability, names=CHANNELS):
    altered = np.asarray(ablated_probability, dtype=float)
    base = np.asarray(original_probability, dtype=float)
    if altered.ndim != 2 or altered.shape[0] != 24 or base.shape != (altered.shape[1],):
        raise ValueError("Sensitivity and baseline arrays disagree")
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Exactly 24 unique sensor/setting labels required")
    if not np.isfinite(altered).all() or not np.isfinite(base).all():
        raise ValueError("Sensitivity results must be finite")
    result = []
    for k, name in enumerate(names):
        signed = (base - altered[k]) * 100.0
        flips = ((base >= .5) != (altered[k] >= .5))
        result.append({
            "channel": name,
            "mean_absolute_probability_change_percentage_points": float(np.mean(np.abs(signed))),
            "median_absolute_probability_change_percentage_points": float(np.median(np.abs(signed))),
            "mean_signed_probability_change_percentage_points": float(np.mean(signed)),
            "endpoint_warning_decision_flips": int(np.count_nonzero(flips)),
        })
    result.sort(key=lambda d: (-d["mean_absolute_probability_change_percentage_points"], d["channel"]))
    return result


def select_case_observation(case):
    """Choose an actually observed window matching the published case class."""
    category = case.get("category")
    if category not in SELECTED_CATEGORY:
        raise ValueError("Unknown recorded case category")
    history = case.get("sampled_history")
    if not isinstance(history, list) or not history:
        raise ValueError("Case history cannot be empty")
    previous = 0
    for row in history:
        cycle = row.get("cycle")
        if type(cycle) is not int or cycle <= previous:
            raise ValueError("Recorded case cycle must be strictly increasing positive integers")
        previous = cycle
        y, p, pred = row.get("actual_rul"), row.get("warning_probability"), row.get("predicted_rul")
        if (type(y) is not int or y < 0 or not isinstance(p, (int, float))
                or not isinstance(pred, (int, float)) or not np.isfinite([p, pred]).all()
                or not 0 <= p <= 1):
            raise ValueError("Recorded case truth/prediction invalid")
    if category == "caught_risk":
        matches = [r for r in history if r["actual_rul"] <= 30 and r["warning_probability"] >= .5]
        selected = matches[0] if matches else None
    elif category == "missed_risk":
        matches = [r for r in history if r["actual_rul"] <= 30 and r["warning_probability"] < .5]
        selected = matches[-1] if matches else None
    elif category == "premature_alert":
        matches = [r for r in history if r["actual_rul"] > 30 and r["warning_probability"] >= .5]
        selected = matches[0] if matches else None
    else:
        matches = [r for r in history if r["actual_rul"] > 30 and r["warning_probability"] < .5]
        selected = matches[-1] if matches else None
    if selected is None:
        raise ValueError("Recorded case category does not match its observations")
    return selected


def run(data_dir: Path, artifacts: Path, replay_path: Path, output: Path, *, download=False):
    import torch
    from .sequence_integrity import (_load_compact, _verified_file, CHECKPOINT_SHA256,
                                     TEMPORAL_SHA256, verify_frozen_reproduction)
    from .longitudinal_replay import make_history
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, ref_path = artifacts / "temporal_qa_checkpoint.pt", artifacts / "temporal_qa_predictions.csv"
    _verified_file(checkpoint, CHECKPOINT_SHA256)
    _verified_file(ref_path, TEMPORAL_SHA256)
    if download:
        download_dataset(data_dir)
    _, test, offsets = load_fd001(data_dir)
    x, y, meta = windows(test, set(test.unit.unique()), official_rul=offsets)
    if len(meta) != 100 or len({unit for unit, cycle in meta}) != 100:
        raise ValueError("Expected 100 distinct official FD001 test engines")
    model, norm = _load_compact(checkpoint)
    mean, _ = norm
    predicted_rul, original_warning, original_risk = predict(model, x, norm)
    verify_frozen_reproduction(pd.read_csv(ref_path), meta, y,
                               predicted_rul, original_warning, original_risk)
    risk_fn = lambda xx: predict(model, xx, norm)[2]
    endpoints_after = ablate_channels(x, mean, original_risk, risk_fn)
    prior = json.loads(replay_path.read_text(encoding="utf-8"))
    if (prior.get("metrics", {}).get("engines", {}).get("total") != 100 or
            not prior.get("original_endpoint_predictions_reproduced") or
            not prior.get("heldout_frozen_model_no_retraining") or
            prior.get("sample_stride_cycles") != 5):
        raise ValueError("Recorded replay integrity contract mismatch")
    cases = prior.get("cases")
    if not isinstance(cases, list) or len(cases) < 4:
        raise ValueError("Missing recorded case examples")
    if len({case.get("engine_id") for case in cases}) != len(cases):
        raise ValueError("Duplicate recorded case engines")
    selected = [select_case_observation(case) for case in cases]
    ids = {int(case["engine_id"]) for case in cases}
    history_x, history_meta = make_history(test[test.unit.isin(ids)], offsets, stride=5)
    positions = {(int(row.unit), int(row.cycle)): i for i, row in history_meta.iterrows()}
    chosen = []
    for case, obs in zip(cases, selected):
        key = (int(case["engine_id"]), int(obs["cycle"]))
        if key not in positions:
            raise ValueError("Recorded case missing from source engine history")
        i = positions[key]
        if int(history_meta.iloc[i].actual_rul) != obs["actual_rul"]:
            raise ValueError("Recorded case truth does not match NASA source")
        chosen.append(i)
    case_x = history_x[chosen].copy()
    case_rul, _, case_risk = predict(model, case_x, norm)
    for k, obs in enumerate(selected):
        if abs(float(case_risk[k]) - float(obs["warning_probability"])) > 1e-4:
            raise ValueError("Frozen historical warning probability did not reproduce")
        if abs(float(case_rul[k]) - float(obs["predicted_rul"])) > 0.025:
            raise ValueError("Frozen historical remaining-life prediction did not reproduce")
    cases_after = ablate_channels(case_x, mean, case_risk, risk_fn)
    observations = []
    for k, (case, obs) in enumerate(zip(cases, selected)):
        before = float(case_risk[k])
        effects = []
        for feature, channel in enumerate(CHANNELS):
            after = float(cases_after[feature, k])
            effects.append({
                "channel": channel,
                "risk_after_channel_mean_replacement": after,
                "risk_change_percentage_points": float(100 * (before - after)),
                "warning_decision_flips": bool((before >= .5) != (after >= .5)),
            })
        effects.sort(key=lambda d: (-abs(d["risk_change_percentage_points"]), d["channel"]))
        observations.append({
            "engine_id": int(case["engine_id"]),
            "cycle": int(obs["cycle"]),
            "category": case["category"],
            "actual_rul_cycles_revealed_after_benchmark": int(obs["actual_rul"]),
            "model_rul_prediction_cycles": float(case_rul[k]),
            "model_warning_probability_before": before,
            "most_sensitive_channels": effects[:5],
        })
    report = {
        "project": "MUSITU AXIOM Sentinel",
        "experiment": "Frozen compact-model channel-wise training-mean replacement sensitivity",
        "not_opentime_5b": True,
        "benchmark": "NASA C-MAPSS FD001 simulated, all 100 official held-out endpoints",
        "method": "Replace one full 30-cycle input channel with its training-only mean from the pinned checkpoint; compare frozen warning probabilities, holding all other channels fixed.",
        "frozen_checkpoint_sha256": CHECKPOINT_SHA256,
        "original_predictions_sha256": TEMPORAL_SHA256,
        "frozen_endpoint_predictions_reproduced": True,
        "recorded_case_predictions_reproduced": True,
        "heldout_engines_scored": 100,
        "channels": summarized_channels(endpoints_after, original_risk, CHANNELS),
        "recorded_cases": observations,
        "limits": [
            "Sensitivity to input replacement is NOT verified physical fault diagnosis or causal explanation.",
            "A constant training mean replacement can create out-of-distribution input and break cross-channel physical relationships.",
            "Post-hoc exploration on the same 100 simulated held-out engines, not new independent validation.",
            "Only the independent compact temporal model; NOT the organizer OpenTime 5B checkpoint.",
            "No real aircraft safety, maintenance savings, or unobserved future data claims.",
            "No raw NASA sensor trajectories are included in this aggregate and selected-case JSON.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/frozen_compact"))
    p.add_argument("--replay-report", type=Path, default=Path("docs/evidence/longitudinal_replay.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/sensor_sensitivity.json"))
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    result = run(a.data_dir, a.artifact_dir, a.replay_report, a.output, download=a.download)
    print(json.dumps({"heldout_engines_scored": result["heldout_engines_scored"],
                      "cases": len(result["recorded_cases"]),
                      "top_channels": result["channels"][:5],
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
