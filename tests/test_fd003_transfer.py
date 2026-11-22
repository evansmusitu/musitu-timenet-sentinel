"""Isolated no-network tests for the FD003 cross-fault experiment."""
from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from sentinel import fd003_transfer as transfer
from sentinel.data import COLS
from sentinel.temporal_qa import TASKS, build_model


def _fixture_fd003(root: Path, *, engines: int = 100, cycles: int = 30) -> tuple[Path, Path]:
    rows = []
    for engine in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            rows.append([engine, cycle] + [0.1 * (engine + cycle + k) for k in range(24)])
    df = pd.DataFrame(rows, columns=COLS)
    test = root / "test_FD003.txt"
    rul = root / "RUL_FD003.txt"
    df.to_csv(test, sep=" ", header=False, index=False)
    rul.write_text("".join(f"{n}\n" for n in range(engines)), encoding="utf-8")
    return test, rul


def test_accepts_100_valid_official_endpoints(tmp_path):
    table, labels = _fixture_fd003(tmp_path)
    frame, offsets = transfer._test_cohort(table, labels)
    assert len(frame) == 3000
    assert len(offsets) == 100
    assert offsets[1] == 0 and offsets[100] == 99


def test_rejects_wrong_engine_count(tmp_path):
    table, labels = _fixture_fd003(tmp_path, engines=99)
    with pytest.raises(ValueError, match="100 FD003"):
        transfer._test_cohort(table, labels)


def test_rejects_short_engine_without_silently_dropping_it(tmp_path):
    table, labels = _fixture_fd003(tmp_path)
    df = pd.read_csv(table, sep=r"\s+", header=None)
    df = df.loc[~((df[0] == 23) & (df[1] == 30))]
    df.to_csv(table, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="fewer than 30"):
        transfer._test_cohort(table, labels)


@pytest.mark.parametrize("label,match", [
    ("NaN", "finite nonnegative"),
    ("-1", "finite nonnegative"),
    ("4.5", "finite nonnegative"),
])
def test_rejects_invalid_rul_labels(tmp_path, label, match):
    table, labels = _fixture_fd003(tmp_path)
    lines = labels.read_text().splitlines()
    lines[2] = label
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        transfer._test_cohort(table, labels)


def test_rejects_missing_official_labels(tmp_path):
    table, labels = _fixture_fd003(tmp_path)
    labels.write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label count mismatch"):
        transfer._test_cohort(table, labels)


def _make_archive(root: Path, *, duplicated: bool = False) -> Path:
    archive = root / "CMAPSSData.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("CMAPSSData/test_FD003.txt", "fake test data\n")
        z.writestr("CMAPSSData/RUL_FD003.txt", "10\n")
        if duplicated:
            z.writestr("backups/RUL_FD003.txt", "100\n")
    return archive


def test_archive_pin_is_enforced(tmp_path):
    _make_archive(tmp_path)
    with pytest.raises(ValueError, match="pinned source MD5"):
        transfer._fd003_files(tmp_path)


def test_archive_rejects_duplicate_basename(tmp_path, monkeypatch):
    archive = _make_archive(tmp_path, duplicated=True)
    monkeypatch.setattr(transfer, "SOURCE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="exactly one RUL_FD003"):
        transfer._fd003_files(tmp_path)


def test_archive_only_extracts_two_fixed_paths(tmp_path, monkeypatch):
    archive = _make_archive(tmp_path)
    monkeypatch.setattr(transfer, "SOURCE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    t, r = transfer._fd003_files(tmp_path)
    assert t.parent == tmp_path and r.parent == tmp_path
    assert t.read_text() == "fake test data\n"
    assert r.read_text() == "10\n"


def test_warning_scoring_exact_counts():
    y = np.array([0, 15, 40, 50, 90])
    p = np.array([True, False, True, False, False])
    assert transfer._warning_scores(y, p) == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }


def test_no_nan_predictions_are_accepted():
    with pytest.raises(ValueError, match="100 FD003"):
        transfer._model_scores(np.array([10.]), np.array([np.nan]), np.array([True]))
    with pytest.raises(ValueError, match="non-finite"):
        transfer._model_scores(np.zeros(100), np.full(100, np.nan), np.zeros(100, dtype=bool))


def test_checkpoint_hash_check_rejects_tamper(tmp_path):
    cp = tmp_path / "temporal_qa_checkpoint.pt"
    cp.write_bytes(b"untrusted pickle placeholder")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        transfer._load_compact(cp)


def test_safe_checkpoint_loads_only_pinned_payload(tmp_path, monkeypatch):
    import torch
    torch.manual_seed(42)
    model = build_model()
    cp = tmp_path / "temporal_qa_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "mean": np.zeros((1, 1, 24), dtype="float32"),
                "scale": np.ones((1, 1, 24), dtype="float32"), "questions": TASKS}, cp)
    monkeypatch.setattr(transfer, "CHECKPOINT_SHA256", hashlib.sha256(cp.read_bytes()).hexdigest())
    loaded, (mean, scale) = transfer._load_compact(cp)
    assert mean.shape == (1, 1, 24) and scale.shape == (1, 1, 24)
    assert set(model.state_dict()) == set(loaded.state_dict())


def test_compact_reference_artifact_sha256_is_required(tmp_path):
    p = tmp_path / "original.csv"
    p.write_bytes(b"mismatched\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        transfer._verified_file(p, transfer.BASELINE_SHA256)
