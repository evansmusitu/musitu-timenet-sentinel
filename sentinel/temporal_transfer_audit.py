"""Zero-shot simulated-cohort transfer of the ORIGINAL FD001-trained compact QA checkpoint.

Uses the independently trained compact 48-wide temporal model, NOT the
organizer's OpenTime 5B. Loads one trusted GitHub Actions artifact with
weights_only=True and narrowly allowed NumPy array datatypes; never retrains,
uses test labels to fit, or publishes model weights or NASA rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .benchmark import metrics
from .data import SOURCE_MD5, SOURCE_URL, download_dataset
from .temporal_qa import build_model, predict, windows
from .transfer_audit import COHORTS, _load_test_cohort

REFERENCE_MAE = 19.04361652851105
REFERENCE_F1 = 0.96
CHECKPOINT_WORKFLOW = "https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425"


def _load_trusted_original_checkpoint(path: Path):
    """Limit pickle to state tensors and known NumPy arrays, not arbitrary globals."""
    import torch

    # The original artifact stores NumPy normalization arrays, rather than
    # tensors, so the default weights-only loader requires these explicit types.
    array_module = np._core if hasattr(np, "_core") else np.core
    allowed = [array_module.multiarray._reconstruct, np.ndarray, np.dtype]
    if hasattr(np, "dtypes"):
        allowed.extend([np.dtypes.Float32DType, np.dtypes.Float64DType])
    else:
        allowed.extend([type(np.dtype("float32")), type(np.dtype("float64"))])
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model_state", "mean", "scale", "questions", "answer_tokens"}:
        raise ValueError("Original compact checkpoint has unexpected/missing keys")
    if np.asarray(checkpoint["mean"]).shape != (1, 1, 24):
        raise ValueError("Unexpected normalization mean shape")
    if np.asarray(checkpoint["scale"]).shape != (1, 1, 24):
        raise ValueError("Unexpected normalization scale shape")
    if not np.isfinite(checkpoint["mean"]).all() or not np.isfinite(checkpoint["scale"]).all():
        raise ValueError("Nonfinite compact checkpoint normalization")
    if (np.asarray(checkpoint["scale"]) <= 0).any():
        raise ValueError("Invalid zero or negative scale")
    model = build_model()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, (checkpoint["mean"], checkpoint["scale"])


def _evaluate(model, norm, test, labels) -> dict:
    all_ids = set(int(x) for x in test.unit.unique())
    eligible = set(int(x) for x in test.groupby("unit").size().loc[lambda sizes: sizes >= 30].index)
    if not eligible:
        raise ValueError("No official test engines have the required 30 observed cycles")
    x, y, metadata = windows(test, eligible, official_rul=labels, window=30)
    if len(metadata) != len(eligible):
        raise ValueError("The official test endpoints do not match eligible engines")
    predicted_rul, flagged, risk_probability = predict(model, x, norm)
    actual = np.asarray(y, dtype=float)
    truth = actual <= 30
    flagged = np.asarray(flagged, dtype=bool)
    p = np.asarray(risk_probability, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability from compact warning head")
    confusion = {
        "tp": int(np.count_nonzero(truth & flagged)),
        "fp": int(np.count_nonzero(~truth & flagged)),
        "fn": int(np.count_nonzero(truth & ~flagged)),
        "tn": int(np.count_nonzero(~truth & ~flagged)),
    }
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reference = metrics(actual, predicted_rul)
    return {
        "heldout_test_engines": len(eligible),
        "official_test_engines_total": len(all_ids),
        "excluded_due_to_fewer_than_30_observations": len(all_ids)-len(eligible),
        "rul_mae_cycles": reference["rul_mae_cycles"],
        "rul_rmse_cycles": reference["rul_rmse_cycles"],
        "warning_prevalence": float(np.mean(truth)),
        "trained_warning_head": {"precision": precision, "recall": recall, "f1": f1},
        "confusion": confusion,
        "warning_probability_brier": float(np.mean((p-truth.astype(float))**2)),
    }


def run(data_dir: Path, checkpoint_path: Path, output_dir: Path, *, download=False) -> dict:
    import torch
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if download:
        download_dataset(data_dir)
    archive_path = data_dir / "CMAPSSData.zip"
    if not archive_path.is_file() or hashlib.md5(archive_path.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError("Missing or unverified NASA C-MAPSS source archive")
    model, norm = _load_trusted_original_checkpoint(checkpoint_path)
    cohort_results = {}
    with ZipFile(archive_path) as archive:
        for subset, (conditions, faults) in COHORTS.items():
            test, labels = _load_test_cohort(archive, subset)
            result = _evaluate(model, norm, test, labels)
            result["operating_conditions"] = conditions
            result["fault_modes"] = faults
            cohort_results[subset] = result

    original = cohort_results["FD001"]
    if not np.isclose(original["rul_mae_cycles"], REFERENCE_MAE, rtol=0, atol=0.02):
        raise ValueError("Original compact checkpoint FD001 MAE reproduction failed")
    if not np.isclose(original["trained_warning_head"]["f1"], REFERENCE_F1, rtol=0, atol=0.005):
        raise ValueError("Original compact checkpoint FD001 warning F1 reproduction failed")
    report = {
        "project": "MUSITU AXIOM Sentinel",
        "experiment": "Zero-shot FD001-to-other-C-MAPSS-cohort transfer of original trained compact temporal QA model",
        "model_identity": "Original independent 48-wide prompt-limited temporal QA Transformer; NOT OpenTime 5B",
        "original_checkpoint_evidence": CHECKPOINT_WORKFLOW,
        "original_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source": SOURCE_URL,
        "source_md5": SOURCE_MD5,
        "zero_target_cohort_training": True,
        "no_test_time_hyperparameter_or_threshold_selection": True,
        "trained_warning_probability_threshold": 0.5,
        "official_test_evaluations": cohort_results,
        "reference_fd001_checkpoint_reproduced": True,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "FD002/FD003/FD004 are NASA simulations, not physically distinct OEMs or operational deployments.",
            "The compact model is separate from the actual organizer OpenTime 5B pilot, whose FD001 warning F1 was 0.3243.",
            "A single frozen FD001 checkpoint was evaluated without adapting to new fault modes or multiple operating conditions.",
            "Results are one endpoint per engine with at least 30 observations; short trajectories are explicitly counted as excluded. No validated continuous online warnings are shown.",
            "Do not infer aviation maintenance safety, economic savings or performance on real hardware.",
            "The source archive and model weights are never redistributed by this report.",
        ],
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"compact_cross_cohort_transfer.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:{"n":v["heldout_test_engines"],
                         "f1":round(v["trained_warning_head"]["f1"],4),
                         "mae":round(v["rul_mae_cycles"],4)} for k,v in cohort_results.items()},indent=2), flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/transfer"))
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    run(args.data_dir,args.checkpoint,args.output_dir,download=args.download)


if __name__=="__main__":
    main()
