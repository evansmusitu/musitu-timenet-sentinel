"""Frozen FD001-trained baseline transfer to NASA FD002/3/4; no target-domain fitting.

Evaluation-only analysis of NASA's simulated C-MAPSS source. This is a separate
robustness experiment, not evidence of new hardware manufacturers, operational
safety, or superior OpenTime 5B performance. Never redistribute NASA raw rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .benchmark import metrics
from .data import SOURCE_MD5, SOURCE_URL, download_dataset, read_table
from .features import build_examples, split_engine_ids

REFERENCE_MAE = 19.40247665823967
REFERENCE_F1 = 0.8888888888888888
COHORTS = {
    "FD001": (1, 1),
    "FD002": (6, 1),
    "FD003": (1, 2),
    "FD004": (6, 2),
}


def _archive_members(archive: ZipFile, names: list[str]) -> dict[str, str]:
    """Resolve filenames unambiguously; reject missing or duplicate archive entries."""
    by_basename: dict[str, list[str]] = {name: [] for name in names}
    for info in archive.infolist():
        basename = Path(info.filename).name
        if basename in by_basename:
            if info.file_size > 50 * 1024 * 1024:
                raise ValueError("Unexpectedly large NASA data member")
            by_basename[basename].append(info.filename)
    ambiguous = [name for name, found in by_basename.items() if len(found) != 1]
    if ambiguous:
        raise ValueError("Missing or ambiguous NASA files: " + ", ".join(ambiguous))
    return {name: found[0] for name, found in by_basename.items()}


def _load_test_cohort(archive: ZipFile, cohort: str) -> tuple[pd.DataFrame, dict[int, int]]:
    """Read test telemetry and official RUL offsets, never train data from target cohort."""
    if cohort not in COHORTS:
        raise ValueError("Unsupported C-MAPSS cohort")
    test_name = f"test_{cohort}.txt"
    target_name = f"RUL_{cohort}.txt"
    members = _archive_members(archive, [test_name, target_name])
    # Existing strict reader validates shape, finiteness, cycles and duplicate IDs.
    with TemporaryDirectory() as tmp:
        test_path = Path(tmp) / test_name
        test_path.write_bytes(archive.read(members[test_name]))
        telemetry = read_table(test_path)
    targets = pd.read_csv(archive.open(members[target_name]), sep=r"\s+", header=None)
    targets = targets.dropna(axis=1, how="all")
    engine_ids = sorted(int(x) for x in telemetry.unit.unique())
    if len(targets) != len(engine_ids) or targets.shape[1] != 1:
        raise ValueError(f"{cohort}: official RUL row count does not match test engines")
    y = pd.to_numeric(targets.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError(f"{cohort}: official RUL targets must be finite nonnegative integers")
    return telemetry, dict(zip(engine_ids, y.astype(int).tolist()))


def _evaluate(model, features: list[str], train_empty: pd.DataFrame,
              test: pd.DataFrame, targets: dict[int, int]) -> dict:
    examples = build_examples(train_empty, test, targets)
    total_engines = int(test.unit.nunique())
    eligible = set(int(x) for x in test.groupby("unit").size().loc[lambda sizes: sizes >= 30].index)
    observed = set(int(x) for x in examples.unit.unique())
    if observed != eligible or list(examples[features].columns) != features:
        raise ValueError(f"Unexpected test endpoint or feature contract: {len(observed)} of {total_engines} engines, {len(features)} features")
    if not eligible:
        raise ValueError("No test engines have the required 30-cycle history")
    scores = np.maximum(0, model.predict(examples[features].to_numpy()))
    actual = examples.rul.to_numpy()
    truth = actual <= 30
    warning = scores <= 30
    result = metrics(actual, scores)
    result["confusion"] = {
        "tp": int(np.count_nonzero(truth & warning)),
        "fp": int(np.count_nonzero(~truth & warning)),
        "fn": int(np.count_nonzero(truth & ~warning)),
        "tn": int(np.count_nonzero(~truth & ~warning)),
    }
    result["heldout_test_engines"] = len(examples)
    result["official_test_engines_total"] = total_engines
    result["excluded_due_to_fewer_than_30_observations"] = total_engines - len(examples)
    return result


def run(data_dir: Path, output_dir: Path, *, download: bool = False) -> dict:
    """Train once on FD001 development data, then freeze model across all four TEST sets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if download:
        download_dataset(data_dir)
    archive_path = data_dir / "CMAPSSData.zip"
    if not archive_path.exists():
        raise FileNotFoundError("Checksum-verified source archive is required")
    if hashlib.md5(archive_path.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError("NASA source MD5 mismatch; refusing unverified transfer evaluation")

    # No test subset, target cohort training data, or official RUL offsets is read
    # until the model is fitted with only FD001 development engine trajectories.
    train = read_table(data_dir / "train_FD001.txt")
    dev = build_examples(train)
    train_ids, val_ids = split_engine_ids(dev)
    excluded = {"split_source", "unit", "cycle", "rul", "failure_30"}
    feature_names = [c for c in dev.columns if c not in excluded]
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, l2_regularization=2.0,
        random_state=42, early_stopping=False,
    )
    fit = dev.unit.isin(train_ids)
    model.fit(dev.loc[fit, feature_names].to_numpy(), dev.loc[fit, "rul"].to_numpy())
    results = {}
    with ZipFile(archive_path) as archive:
        for cohort, (operating_conditions, fault_modes) in COHORTS.items():
            test, labels = _load_test_cohort(archive, cohort)
            scores = _evaluate(model, feature_names, train.iloc[:0], test, labels)
            scores["operating_conditions"] = operating_conditions
            scores["fault_modes"] = fault_modes
            results[cohort] = scores

    ref = results["FD001"]
    # Explicit reproduction gate: source and fixed training recipe must match
    # the existing successful public FD001 baseline; otherwise do not present a
    # new result as directly comparable with the original freeze.
    if not np.isclose(ref["rul_mae_cycles"], REFERENCE_MAE, atol=0.02, rtol=0):
        raise ValueError(f"FD001 reference MAE failed reproduction check: {ref['rul_mae_cycles']}")
    if not np.isclose(ref["warning_30_f1"], REFERENCE_F1, atol=0.005, rtol=0):
        raise ValueError("FD001 warning F1 failed reproduction check")

    report = {
        "project": "MUSITU AXIOM Sentinel",
        "experiment": "zero-shot cross-cohort transfer of frozen FD001-trained classical CPU baseline",
        "source_url": SOURCE_URL,
        "source_md5": SOURCE_MD5,
        "model": "HistGradientBoostingRegressor (not OpenTime 5B or the compact QA model)",
        "training_cohort_only": "FD001",
        "training_engine_count": len(train_ids),
        "validation_engine_count": len(val_ids),
        "feature_count": len(feature_names),
        "training_random_state": 42,
        "warning_horizon_cycles": 30,
        "target_cohort_training_or_retraining": False,
        "target_test_used_to_select_parameters": False,
        "reference_fd001_reproduced": True,
        "official_test_evaluations": results,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "All four cohorts are NASA C-MAPSS SIMULATIONS, not distinct physical hardware manufacturers or independent real-world operators.",
            "The target subsets have different fault-mode/operating-condition mixes; model transfer failure is expected and scientifically informative.",
            "No new data was used to fit or tune the frozen FD001 model for FD002, FD003 or FD004.",
            "Only one end-of-observation endpoint is scored for each engine with at least 30 observed cycles; engines with fewer are explicitly counted as excluded; no streaming warning lead times are inferred.",
            "These test results are descriptive and provide no real-aircraft operational safety certification.",
            "No original NASA raw source records or target-engine individual labels are redistributed.",
            "The trained model is a classical CPU baseline, NOT the organizer's 5B TSLM.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "cross_cohort_transfer.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({cohort: {"warning_f1": round(v["warning_30_f1"], 4),
                                "rul_mae": round(v["rul_mae_cycles"], 3),
                                "engines": v["heldout_test_engines"]} for cohort, v in results.items()},
                     indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transfer"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, download=args.download)


if __name__ == "__main__":
    main()
