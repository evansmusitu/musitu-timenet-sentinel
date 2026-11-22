"""CPU-only sequence integrity audit of an existing, independently trained model.

Not an OpenTime checkpoint experiment. No training, tuning, or test threshold
selection happens here. Each perturbation changes only past sensor inputs.
Ablations measure model sensitivity, not trustworthy causal explanations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_dataset, load_fd001
from .temporal_qa import TASKS, build_model, predict, windows

# Pinned successful November 20, 2026 GitHub Actions frozen artifacts.
CHECKPOINT_SHA256 = "6f085be7015a48d48d6c1fed28dae9e888ee19dec495fcba085f375edbc6847c"
TEMPORAL_SHA256 = "bbc89e6a15a5fca61165466808202d0a5924aa55218660b8c581ed6022f890ec"


def _verified_file(path: Path, digest: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen artifact SHA256 mismatch: {path.name}")


def _load_compact(checkpoint: Path):
    import torch
    from numpy._core.multiarray import _reconstruct
    _verified_file(checkpoint, CHECKPOINT_SHA256)
    # Trust only narrowly allowlisted NumPy types used in the pinned checkpoint.
    # NEVER load a user-supplied or unpinned checkpoint with weights_only=False.
    safe = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("questions") != TASKS:
        raise ValueError("Frozen checkpoint question contract mismatch")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (1, 1, 24) or scale.shape != (1, 1, 24):
        raise ValueError("Frozen checkpoint normalization shape mismatch")
    if (not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Frozen checkpoint normalization invalid")
    model = build_model()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, (mean, scale)


SEED = 20261122
CONDITIONS = ("original", "reverse_time", "shuffle_time", "last_value_only", "mean_value_only")


def perturb(x: np.ndarray, condition: str, *, seed: int = SEED) -> np.ndarray:
    """Return a transformed copy of [engine, past_cycle, 24 sensor/settings]."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown perturbation: {condition}")
    arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[2] != 24 or arr.shape[1] < 2 or arr.shape[0] < 1:
        raise ValueError("Expected nonempty (N, time>=2, 24 channels) telemetry")
    if not np.isfinite(arr).all():
        raise ValueError("Telemetry must be finite")
    if condition == "original":
        return arr.copy()
    if condition == "reverse_time":
        return arr[:, ::-1, :].copy()
    if condition == "last_value_only":
        return np.repeat(arr[:, -1:, :], arr.shape[1], axis=1)
    if condition == "mean_value_only":
        return np.repeat(arr.mean(axis=1, keepdims=True), arr.shape[1], axis=1)
    rng = np.random.default_rng(seed)
    # One coherent time reordering per engine. Do not shuffle channels separately,
    # because that would also destroy same-time cross-channel relationships.
    out = arr.copy()
    for engine in range(arr.shape[0]):
        out[engine] = arr[engine, rng.permutation(arr.shape[1]), :]
    return out


def verify_frozen_reproduction(
    reference: pd.DataFrame, meta: list[tuple[int, int]], actual: np.ndarray,
    predicted: np.ndarray, warnings: np.ndarray, probabilities: np.ndarray,
) -> None:
    """Fail before ablations if the original prediction artifact is not reproduced."""
    required = {"unit", "observed_cycle", "actual_rul", "predicted_rul",
                "risk_within_30_predicted", "risk_within_30_probability"}
    if required.difference(reference.columns):
        raise ValueError("Frozen reference missing required fields")
    if len(reference) != len(meta) or len(reference) != 100:
        # Allow small synthetic fixtures in dedicated verification unit tests.
        if len(reference) != len(meta) or len(reference) < 1:
            raise ValueError("Frozen reference engine count mismatch")
    if len(set((int(unit), int(cycle)) for unit, cycle in meta)) != len(meta):
        raise ValueError("Duplicate frozen reference engine/cycle")
    if (reference["unit"].tolist() != [u for u, _ in meta] or
            reference["observed_cycle"].tolist() != [c for _, c in meta]):
        raise ValueError("Frozen reference engine/cycle mismatch or reordered engines")
    truth = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    warnings = np.asarray(warnings, dtype=bool)
    if not (len(truth) == len(p) == len(probability) == len(warnings) == len(meta)):
        raise ValueError("Frozen reference vector lengths disagree")
    if not all(np.isfinite(v).all() for v in (truth, p, probability)):
        raise ValueError("Frozen reference vectors include nonfinite values")
    if not np.array_equal(truth, reference.actual_rul.to_numpy(dtype=float)):
        raise ValueError("Frozen reference ground truth does not match")
    if not np.allclose(p, reference.predicted_rul.to_numpy(dtype=float), rtol=0, atol=1e-4):
        raise ValueError("Frozen reference RUL predictions do not match")
    if not np.allclose(probability, reference.risk_within_30_probability.to_numpy(dtype=float), rtol=0, atol=1e-5):
        raise ValueError("Frozen reference probabilities do not match")
    raw_flags = reference.risk_within_30_predicted
    if raw_flags.dtype != bool and not raw_flags.astype(str).str.lower().isin(("true", "false")).all():
        raise ValueError("Frozen reference warnings not boolean")
    flags = raw_flags.astype(str).str.lower().eq("true").to_numpy(dtype=bool)
    if not np.array_equal(warnings, flags):
        raise ValueError("Frozen reference warning decisions differ")


def score_condition(actual: np.ndarray, predictions: np.ndarray, warnings: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predictions, dtype=float)
    w = np.asarray(warnings, dtype=bool)
    q = np.asarray(probabilities, dtype=float)
    if not (len(y) == len(p) == len(w) == len(q)) or not len(y):
        raise ValueError("Score input lengths disagree or empty")
    if any(not np.isfinite(v).all() for v in (y, p, q)) or (q < 0).any() or (q > 1).any():
        raise ValueError("Score input contains nonfinite or invalid probability")
    truth = y <= 30
    tp = int((truth & w).sum())
    fp = int((~truth & w).sum())
    fn = int((truth & ~w).sum())
    tn = int((~truth & ~w).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "warning_precision": precision, "warning_recall": recall,
        "warning_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "rul_mae_cycles": float(np.abs(y - p).mean()),
        "warning_probability_brier_score": float(((q - truth.astype(float))**2).mean()),
    }


def run(data_dir: Path, artifacts: Path, output: Path, *, download: bool = False) -> dict:
    """Audit a frozen compact checkpoint on original FD001 test engines."""
    import torch
    torch.set_num_threads(min(2, torch.get_num_threads()))
    _verified_file(artifacts / "temporal_qa_checkpoint.pt", CHECKPOINT_SHA256)
    _verified_file(artifacts / "temporal_qa_predictions.csv", TEMPORAL_SHA256)
    if download:
        download_dataset(data_dir)
    _, test, rul = load_fd001(data_dir)
    x, y, meta = windows(test, set(test.unit.unique()), official_rul=rul)
    if len(meta) != 100:
        raise ValueError("Expected exactly 100 official FD001 test endpoints")
    model, norm = _load_compact(artifacts / "temporal_qa_checkpoint.pt")
    reference = pd.read_csv(artifacts / "temporal_qa_predictions.csv")
    report = {
        "experiment": "Frozen compact temporal model, sensor time-order ablations",
        "not_opentime_5b": True,
        "benchmark": "NASA C-MAPSS FD001 simulated, 100 final test endpoints",
        "threshold_cycles": 30,
        "warning_probability_threshold": 0.5,
        "model_checkpoint_sha256": CHECKPOINT_SHA256,
        "original_predictions_sha256": TEMPORAL_SHA256,
        "seed": SEED,
        "conditions": {},
        "notes": [
            "Exploratory post-hoc ablations; no retraining, calibration or threshold tuning.",
            "Reference FD001 model predictions must reproduce before any perturbation is scored.",
            "Time-order manipulations preserve each sensor channel's values; shuffle uses a shared time permutation across channels per engine.",
            "Ablation score changes indicate sensitivity, not verified natural-language explanation or causality.",
            "Model is an independently trained compact architecture; NOT the organizer's OpenTime 5B checkpoint.",
            "No per-engine NASA observations or labels are published in this aggregate JSON.",
        ],
    }
    for condition in CONDITIONS:
        p, w, q = predict(model, perturb(x, condition), norm)
        if condition == "original":
            verify_frozen_reproduction(reference, meta, y, p, w, q)
        report["conditions"][condition] = score_condition(y, p, w, q)
    original = report["conditions"]["original"]
    for condition, row in report["conditions"].items():
        row["delta_warning_f1_from_original"] = row["warning_f1"] - original["warning_f1"]
        row["delta_rul_mae_from_original_cycles"] = row["rul_mae_cycles"] - original["rul_mae_cycles"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/frozen_compact"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sequence_integrity.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.artifact_dir, args.output, download=args.download), indent=2))


if __name__ == "__main__":
    main()
