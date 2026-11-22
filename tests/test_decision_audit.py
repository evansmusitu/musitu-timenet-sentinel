"""Checks for the immutable *held-out* paired audit: no post-test fitting."""
from __future__ import annotations

import json
import pandas as pd
import pytest

from sentinel.decision_audit import evaluate, load_paired_artifacts, run


def records(tmp_path):
    base = pd.DataFrame({
        "unit": [3, 1, 2, 4],
        "cycle": [41, 31, 32, 33],
        "rul": [15, 5, 60, 40],
        "predicted_rul": [20, 50, 60, 50],
        "warning_30": [True, False, False, False],
    })
    temp = pd.DataFrame({
        "unit": [4, 2, 1, 3],
        "observed_cycle": [33, 32, 31, 41],
        "actual_rul": [40, 60, 5, 15],
        "predicted_rul": [44, 58, 8, 14],
        "risk_within_30_probability": [.2, .01, .9, .8],
        "risk_within_30_predicted": [False, False, True, True],
    })
    a, b = tmp_path / "base.csv", tmp_path / "compact.csv"
    base.to_csv(a, index=False)
    temp.to_csv(b, index=False)
    return a, b


def test_pairing_uses_unit_and_reconciles_labels(tmp_path):
    a, b = records(tmp_path)
    paired = load_paired_artifacts(a, b, expected_engines=4)
    assert paired.unit.tolist() == [1, 2, 3, 4]
    assert paired.rul.tolist() == [5, 60, 15, 40]


def test_confusion_and_aggregate_only_output(tmp_path):
    a, b = records(tmp_path)
    report = run(a, b, tmp_path / "audit.json", samples=250, expected_engines=4)
    assert report["baseline"]["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert report["compact_temporal_qa"]["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert report["paired_observed_difference"]["additional_true_warnings"] == 1
    assert report["paired_observed_difference"]["additional_false_alarms"] == 0
    assert report["test_population"]["independent_engine_endpoints"] == 4
    assert "baseline_predictions_sha256" in report["provenance"]
    assert report["illustrative_penalty_scenarios"][0]["baseline_penalty_units"] == 1
    published = json.loads((tmp_path / "audit.json").read_text())
    assert "unit" not in published and "per_engine" not in published
    assert published["bootstrap_uncertainty"]["resamples"] == 250


def test_bootstrap_seed_is_deterministic(tmp_path):
    a, b = records(tmp_path)
    paired = load_paired_artifacts(a, b, expected_engines=4)
    assert evaluate(paired, samples=250) == evaluate(paired, samples=250)


@pytest.mark.parametrize("file,field,value,match", [
    ("baseline", "unit", 1, "engine IDs"),
    ("baseline", "rul", float("nan"), "non-finite"),
    ("baseline", "rul", 999, "labels disagree"),
    ("baseline", "cycle", -1, "cannot be negative"),
    ("baseline", "warning_30", "maybe", "true/false"),
    ("temporal", "observed_cycle", 1000, "observation cycles disagree"),
    ("temporal", "risk_within_30_probability", 2.0, "probabilities"),
    ("temporal", "risk_within_30_predicted", "true", "warning flag disagrees"),
])
def test_rejects_bad_model_artifact(tmp_path, file, field, value, match):
    a, b = records(tmp_path)
    path = a if file == "baseline" else b
    frame = pd.read_csv(path)
    if field in ("warning_30", "risk_within_30_predicted"):
        frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_paired_artifacts(a, b, expected_engines=4)


def test_rejects_partial_and_unmatched_engine_sets(tmp_path):
    a, b = records(tmp_path)
    temp = pd.read_csv(b)
    temp.loc[0, "unit"] = 999
    temp.to_csv(b, index=False)
    with pytest.raises(ValueError, match="engine ID sets disagree"):
        load_paired_artifacts(a, b, expected_engines=4)


def test_rejects_too_few_bootstrap_samples(tmp_path):
    a, b = records(tmp_path)
    paired = load_paired_artifacts(a, b, expected_engines=4)
    with pytest.raises(ValueError, match="At least 100"):
        evaluate(paired, samples=50)
