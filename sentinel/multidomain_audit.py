"""Exploratory multi-domain C-MAPSS training with FD004 held-out combination.

This is a post-hoc response to the observed failure of FD001-only transfer.
It uses FD001/FD002/FD003 DEVELOPMENT source files for training and keeps
FD004 entirely out of fit, validation and architecture/threshold selection.
It evaluates on official FD004 TEST once and publishes only aggregate metrics.
The data and models are simulated research, never real aircraft maintenance.
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

from .data import SOURCE_URL, SOURCE_MD5, download_dataset, read_table
from .benchmark import metrics
from .features import build_examples, split_engine_ids
from .transfer_audit import _archive_members, _load_test_cohort

TRAIN_COHORTS = ("FD001", "FD002", "FD003")
TARGET_COHORT = "FD004"
RESERVED = {"split_source", "unit", "cycle", "rul", "failure_30"}


def _load_training_source(archive: ZipFile, cohort: str) -> pd.DataFrame:
    """Load *training* run-to-failure trajectories, never target test data."""
    if cohort not in TRAIN_COHORTS:
        raise ValueError("FD004 target cohort is forbidden as a training source")
    name = f"train_{cohort}.txt"
    member = _archive_members(archive, [name])[name]
    with TemporaryDirectory() as td:
        path = Path(td)/name
        path.write_bytes(archive.read(member))
        return read_table(path)


def _split_one_cohort(cohort: str, source: pd.DataFrame):
    examples = build_examples(source)
    if examples.empty:
        raise ValueError(f"{cohort}: no 30-cycle development windows")
    train_ids, val_ids = split_engine_ids(examples, seed=42)
    if not set(train_ids).isdisjoint(val_ids):
        raise ValueError("An engine occurs in both development partitions")
    train = examples.loc[examples.unit.isin(train_ids)].copy()
    validation = examples.loc[examples.unit.isin(val_ids)].copy()
    if train.empty or validation.empty:
        raise ValueError("Source cohort lacks an independent validation partition")
    return train, validation, {"cohort": cohort, "train_engines": len(train_ids),
                               "validation_engines": len(val_ids),
                               "train_windows": len(train), "validation_windows": len(validation)}


def _evaluate(model, test: pd.DataFrame, labels: dict[int,int], feature_names: list[str]) -> dict:
    empty_source = test.iloc[:0]
    examples = build_examples(empty_source, test, labels)
    all_engines = int(test.unit.nunique())
    eligible = set(int(x) for x in test.groupby("unit").size().loc[lambda sizes: sizes >= 30].index)
    observed = set(int(x) for x in examples.unit.unique())
    if observed != eligible or not eligible:
        raise ValueError("Invalid eligible FD004 endpoint selection")
    if set(examples.columns) - RESERVED != set(feature_names):
        raise ValueError("FD004 feature contract differs from training sources")
    predicted = np.maximum(0.0, model.predict(examples[feature_names].to_numpy()))
    actual = examples.rul.to_numpy()
    warning = predicted <= 30
    truth = actual <= 30
    cm = {
        "tp": int((truth & warning).sum()),
        "fp": int((~truth & warning).sum()),
        "fn": int((truth & ~warning).sum()),
        "tn": int((~truth & ~warning).sum()),
    }
    result = metrics(actual, predicted)
    result.update({"confusion": cm, "official_test_engines_total":all_engines,
                   "heldout_test_engines": len(examples),
                   "excluded_due_to_fewer_than_30_observations": all_engines-len(examples)})
    return result


def run(data_dir: Path, output_dir: Path, *, download=False) -> dict:
    if download:
        download_dataset(data_dir)
    archive_path = data_dir / "CMAPSSData.zip"
    if not archive_path.exists() or hashlib.md5(archive_path.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError("NASA archive absent or checksum mismatch")

    source_train, source_val, counts = [], [], []
    feature_contract = None
    with ZipFile(archive_path) as archive:
        for cohort in TRAIN_COHORTS:
            training_source = _load_training_source(archive, cohort)
            train, val, provenance = _split_one_cohort(cohort, training_source)
            features = [col for col in train.columns if col not in RESERVED]
            if feature_contract is None:
                feature_contract = features
            if feature_contract != features:
                raise ValueError("Training feature contracts mismatch across NASA cohorts")
            source_train.append(train)
            source_val.append(val)
            counts.append(provenance)
        # The target official test was deliberately not read until the full
        # training/validation feature matrix was constructed.
        train_set = pd.concat(source_train, ignore_index=True)
        val_set = pd.concat(source_val, ignore_index=True)
        model = HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=15, l2_regularization=2.0,
            random_state=42, early_stopping=False,
        )
        model.fit(train_set[feature_contract].to_numpy(), train_set.rul.to_numpy())
        val_pred = np.maximum(0.0, model.predict(val_set[feature_contract].to_numpy()))
        val_scores = metrics(val_set.rul.to_numpy(), val_pred)
        target, target_labels = _load_test_cohort(archive, TARGET_COHORT)
        result = _evaluate(model, target, target_labels, feature_contract)

    report = {
        "project":"MUSITU AXIOM Sentinel",
        "experiment":"exploratory leave-FD004-combination-out simulated-domain generalization",
        "motivation":"Post-hoc response to measured zero-shot FD001-to-FD004 transfer failure",
        "source_url": SOURCE_URL, "source_md5": SOURCE_MD5,
        "training_cohorts":list(TRAIN_COHORTS),
        "training_cohort_counts":counts,
        "heldout_target_cohort": TARGET_COHORT,
        "target_operating_conditions": 6,
        "target_fault_modes": 2,
        "fd004_data_used_in_fit_or_validation": False,
        "feature_count":len(feature_contract),
        "model":"HistGradientBoostingRegressor classical CPU baseline; NOT OpenTime 5B",
        "fixed_hyperparameters":{"max_iter":180,"max_leaf_nodes":15,"l2_regularization":2.0,"random_state":42,
                                  "early_stopping":False,"warning_horizon_cycles":30},
        "independent_development_validation":val_scores,
        "official_fd004_heldout_test":result,
        "evaluated_utc":datetime.now(timezone.utc).isoformat(),
        "limitations":[
            "This experiment was conceived after observing poor FD001-only transfer; it is exploratory, not preregistered.",
            "Although FD004 was entirely withheld from training, this is not real-world OEM transfer: all cohorts share NASA's simulated engine family.",
            "Only target engines with >=30 cycles are scored; excluded short histories are separately counted.",
            "The single operating threshold of 30 cycles and model hyperparameters were fixed before this experiment's FD004 evaluation.",
            "Engine ID grouping is applied within each source cohort; numeric engine identifiers across different datasets are unrelated.",
            "We cannot infer operational maintenance savings, aviation safety, or continuous streaming warning lead times.",
            "No raw NASA dataset rows, individual engine RUL labels or trained model weights are republished.",
        ]
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"multidomain_fd004_aggregate.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"independent_development_validation": {"mae":round(val_scores["rul_mae_cycles"],4),"f1":round(val_scores["warning_30_f1"],4)},
                      "heldout_fd004":{"mae":round(result["rul_mae_cycles"],4),"f1":round(result["warning_30_f1"],4),
                                        "tested":result["heldout_test_engines"],"excluded_short":result["excluded_due_to_fewer_than_30_observations"]}},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/multidomain"))
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,download=args.download)

if __name__=="__main__":
    main()
