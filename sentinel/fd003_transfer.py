"""Frozen FD001-to-FD003 cross-fault evaluation of Sentinel's *separate CPU* models.

This experiment is precommitted to evaluate the official 100-engine FD003 test
set exactly once. No FD003 train records or labels influence model parameters,
feature scaling, warning thresholds, or model selection.

The compact model is NOT the organizer's OpenTime 5B checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .benchmark import metrics
from .data import SOURCE_MD5, download_dataset, load_fd001, read_table
from .features import build_examples, split_engine_ids
from .temporal_qa import TASKS, build_model, predict, windows

# Original artifacts from the Nov 20, 2026 successful, independent FD001 runs.
BASELINE_SHA256 = "eb6ba0cbc4007ea32c8da1ce27b8e4818ae3a53f217129975aa9c638bf5cccd3"
TEMPORAL_SHA256 = "bbc89e6a15a5fca61165466808202d0a5924aa55218660b8c581ed6022f890ec"
CHECKPOINT_SHA256 = "6f085be7015a48d48d6c1fed28dae9e888ee19dec495fcba085f375edbc6847c"
SOURCE_RUNS = {
    "classical_fd001": "https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482",
    "compact_fd001": "https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425",
}
WARN_CYCLES = 30


def _verified_file(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Frozen artifact SHA256 mismatch: {path.name}")


def _fd003_files(root: Path) -> tuple[Path, Path]:
    """Extract only checksum-authenticated FD003 test files, never train_FD003."""
    archive = root / "CMAPSSData.zip"
    if hashlib.md5(archive.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError("NASA archive does not match the pinned source MD5")
    needed = ("test_FD003.txt", "RUL_FD003.txt")
    paths = []
    with ZipFile(archive) as z:
        for basename in needed:
            matches = [member for member in z.infolist() if Path(member.filename).name == basename]
            if len(matches) != 1:
                raise ValueError(f"NASA archive must contain exactly one {basename}")
            member = matches[0]
            if member.is_dir() or member.file_size > 25 * 1024 * 1024:
                raise ValueError(f"Unreasonable FD003 source file size for {basename}")
            path = root / basename
            # Write to a fixed basename: never use untrusted ZIP path components.
            path.write_bytes(z.read(member))
            paths.append(path)
    return paths[0], paths[1]


def _test_cohort(table: Path, rul_table: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    frame = read_table(table)
    ids = sorted(frame.unit.unique())
    if len(ids) != 100:
        raise ValueError("Expected 100 FD003 official test engines")
    if not np.array_equal(ids, np.arange(1, 101)):
        raise ValueError("FD003 official test engine IDs must be 1..100")
    if any(len(group) < 30 for _, group in frame.groupby("unit")):
        raise ValueError("FD003 test engine has fewer than 30 observed cycles")
    offsets = pd.read_csv(rul_table, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    if offsets.shape != (len(ids), 1):
        raise ValueError("FD003 official RUL label count mismatch")
    vals = pd.to_numeric(offsets[0], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any() or not np.array_equal(vals, np.floor(vals)):
        raise ValueError("FD003 official RUL labels must be finite nonnegative integers")
    return frame, dict(zip(ids, vals.astype(int).tolist(), strict=True))


def _reference(actual: pd.DataFrame, baseline: Path, temporal: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enforce original 100-engine frozen predictions and labels before FD003."""
    _verified_file(baseline, BASELINE_SHA256)
    _verified_file(temporal, TEMPORAL_SHA256)
    base = pd.read_csv(baseline).sort_values("unit").reset_index(drop=True)
    qa = pd.read_csv(temporal).sort_values("unit").reset_index(drop=True)
    actual = actual.sort_values("unit").reset_index(drop=True)
    if len(actual) != 100 or len(base) != 100 or len(qa) != 100:
        raise ValueError("Frozen FD001 reference count must equal 100")
    if not np.array_equal(actual.unit, base.unit) or not np.array_equal(actual.unit, qa.unit):
        raise ValueError("Frozen FD001 reference engine IDs disagree")
    if not np.array_equal(actual.cycle, base.cycle) or not np.array_equal(actual.cycle, qa.observed_cycle):
        raise ValueError("Frozen FD001 reference observation cycles disagree")
    if not np.array_equal(actual.rul, base.rul) or not np.array_equal(actual.rul, qa.actual_rul):
        raise ValueError("Frozen FD001 reference official labels disagree")
    return base, qa


def _warning_scores(y: np.ndarray, warning: np.ndarray) -> dict:
    truth = y <= WARN_CYCLES
    prediction = warning.astype(bool)
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def _model_scores(y: np.ndarray, estimated_rul: np.ndarray, warning: np.ndarray) -> dict:
    if len(y) != 100 or len(estimated_rul) != 100 or len(warning) != 100:
        raise ValueError("Expected exactly 100 FD003 endpoints")
    if not np.isfinite(estimated_rul).all():
        raise ValueError("Model emitted non-finite remaining-life predictions")
    return {
        "rul_mae_cycles": float(np.abs(y - estimated_rul).mean()),
        "rul_rmse_cycles": float(np.sqrt(((y - estimated_rul) ** 2).mean())),
        "warning_30": _warning_scores(y, warning),
    }


def _fit_classical(fd001_train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, list[str]]:
    # Exact frozen settings and FD001 development split from sentinel.benchmark.
    all_rows = build_examples(fd001_train)
    train_ids, _ = split_engine_ids(all_rows)
    train_sel = all_rows.unit.isin(train_ids)
    reserved = {"split_source", "unit", "cycle", "rul", "failure_30"}
    cols = [column for column in all_rows.columns if column not in reserved]
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, l2_regularization=2.0,
        random_state=42, early_stopping=False,
    )
    model.fit(all_rows.loc[train_sel, cols].to_numpy(), all_rows.loc[train_sel, "rul"].to_numpy())
    return model, cols


def _predict_classical(model: HistGradientBoostingRegressor, features: list[str], data: pd.DataFrame,
                       labels: dict[int, int]) -> tuple[pd.DataFrame, np.ndarray]:
    endpoints = build_examples(data.iloc[0:0], data, labels).sort_values("unit").reset_index(drop=True)
    if len(endpoints) != 100:
        raise ValueError("Expected exactly one prediction per official test engine")
    if any(col not in endpoints for col in features):
        raise ValueError("Feature schema drift between FD001 and FD003")
    pred = np.maximum(0, model.predict(endpoints[features].to_numpy()))
    return endpoints, pred


def _load_compact(checkpoint: Path):
    import torch
    from numpy._core.multiarray import _reconstruct
    _verified_file(checkpoint, CHECKPOINT_SHA256)
    # The pinned checkpoint serializes numpy normalization arrays. Explicitly
    # allow only the NumPy types required by weights_only=True; never unpickle
    # an unpinned checkpoint with weights_only=False.
    safe = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("questions") != TASKS:
        raise ValueError("Compact checkpoint prompt contract disagrees")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (1, 1, 24) or scale.shape != (1, 1, 24):
        raise ValueError("Compact checkpoint normalization shape disagrees")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid compact checkpoint normalization")
    model = build_model()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, (mean, scale)


def run(data_dir: Path, artifact_dir: Path, output: Path, *, download: bool = False) -> dict:
    """Produce only aggregate, labeled FD003 test metrics; never fit on FD003."""
    if download:
        download_dataset(data_dir)
    fd001_train, fd001_test, fd001_labels = load_fd001(data_dir)
    # Explicit same-archive authentication for the FD003 source.
    fd003_file, fd003_rul_file = _fd003_files(data_dir)
    fd003_test, fd003_labels = _test_cohort(fd003_file, fd003_rul_file)

    classical, features = _fit_classical(fd001_train)
    original_rows, original_pred = _predict_classical(classical, features, fd001_test, fd001_labels)
    base_ref, compact_ref = _reference(
        original_rows,
        artifact_dir / "official_test_predictions.csv",
        artifact_dir / "temporal_qa_predictions.csv",
    )
    if not np.allclose(original_pred, base_ref.predicted_rul.to_numpy(), atol=1e-4, rtol=0):
        raise ValueError("Rebuilt FD001 baseline fails frozen original prediction verification")
    if not np.array_equal(original_pred <= WARN_CYCLES, base_ref.warning_30.to_numpy(dtype=bool)):
        raise ValueError("Rebuilt FD001 baseline warning flags differ")

    model, norm = _load_compact(artifact_dir / "temporal_qa_checkpoint.pt")
    x001, y001, meta001 = windows(fd001_test, set(fd001_test.unit.unique()), official_rul=fd001_labels)
    p001, w001, prob001 = predict(model, x001, norm)
    if [row[0] for row in meta001] != compact_ref.unit.tolist():
        raise ValueError("Rebuilt FD001 compact engine order disagrees")
    if not np.array_equal(y001, compact_ref.actual_rul.to_numpy()):
        raise ValueError("Rebuilt FD001 compact official labels disagree")
    if not np.allclose(p001, compact_ref.predicted_rul.to_numpy(), atol=1e-4, rtol=0):
        raise ValueError("Rebuilt FD001 compact RUL predictions disagree")
    if not np.allclose(prob001, compact_ref.risk_within_30_probability.to_numpy(), atol=1e-5, rtol=0):
        raise ValueError("Rebuilt FD001 compact warning probabilities disagree")
    if not np.array_equal(w001, compact_ref.risk_within_30_predicted.to_numpy(dtype=bool)):
        raise ValueError("Rebuilt FD001 compact warning flags disagree")

    target, pred_classical = _predict_classical(classical, features, fd003_test, fd003_labels)
    x003, y003, meta003 = windows(fd003_test, set(fd003_test.unit.unique()), official_rul=fd003_labels)
    if not np.array_equal(target.unit, np.array([unit for unit, _ in meta003])) or not np.array_equal(target.rul, y003):
        raise ValueError("FD003 evaluation labels or engine order disagree across model families")
    pred_compact, warn_compact, prob_compact = predict(model, x003, norm)
    baseline = _model_scores(y003, pred_classical, pred_classical <= WARN_CYCLES)
    compact = _model_scores(y003, pred_compact, warn_compact)
    compact["risk_probability_brier_score"] = float(np.mean((prob_compact - (y003 <= WARN_CYCLES)) ** 2))
    result = {
        "experiment": "FD001-trained models tested without adaptation on FD003 two-fault-mode simulated engines",
        "dataset": {"training": "NASA C-MAPSS FD001", "unseen_evaluation": "NASA C-MAPSS FD003", "official_test_engines": 100,
                    "source_archive_md5": SOURCE_MD5},
        "methods": {"training": "FD001 development engines only", "FD003_training_used": False,
                    "FD003_threshold_tuning_used": False, "warning_horizon_cycles": WARN_CYCLES,
                    "compact_model_is_OpenTime_5B": False,
                    "FD001_predictions_verified_before_FD003_evaluation": True},
        "FD003": {"classical_baseline": baseline, "compact_temporal_qa": compact},
        "reference_FD001": {"classical_baseline": _model_scores(original_rows.rul.to_numpy(), original_pred, original_pred <= WARN_CYCLES),
                           "compact_temporal_qa": _model_scores(y001, p001, w001)},
        "artifact_sha256": {"baseline_predictions": BASELINE_SHA256, "compact_predictions": TEMPORAL_SHA256,
                            "compact_checkpoint": CHECKPOINT_SHA256},
        "source_workflows": SOURCE_RUNS,
        "limitations": [
            "Both evaluated models are independent of the organizer's OpenTime 5B checkpoint.",
            "C-MAPSS data are simulated, not live equipment; FD003 shares the simulator with FD001.",
            "Only one endpoint per official FD003 test engine is evaluated; not streaming alert lead time.",
            "No FD003 training or validation is performed; failures on new fault modes remain possible.",
            "FD003 individual fault-mode identities are not provided; per-fault performance cannot be asserted.",
            "The downstream FD001 comparison is confirmed against frozen artifact SHA256 hashes.",
            "No raw NASA source rows, labels, or checkpoint weights are redistributed in this report.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fd003_transfer_aggregate.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.artifact_dir, args.output, download=args.download), indent=2))


if __name__ == "__main__":
    main()
