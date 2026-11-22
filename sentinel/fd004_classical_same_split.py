"""Like-for-like classical baseline for the OpenTime FD004 run (CPU only).

The existing classical FD004 result (F1 0.816) was trained on FD001-FD003 and never saw FD004.
The OpenTime FD004 run trains on FD004 development engines, so this script gives the classical
classifier the same advantage: the same engine-disjoint FD004 split, the same per-condition
normalisation from development-train engines, the repository's standard 30-cycle window features
plus engine age, the same HistGradientBoosting settings and the same 0.1-0.9 validation threshold
grid as `multidomain_classifier.py`. The official FD004 test set is scored once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import SOURCE_MD5, download_dataset
from .features import build_examples
from .opentime_examples import partitions
from .opentime_fd004 import load_fd004
from .opentime_v2_examples import confusion

THRESHOLDS = tuple(float(x) for x in np.linspace(0.1, 0.9, 17))
RESERVED = {"split_source", "unit", "cycle", "rul", "failure_30"}


def run(root: Path, output: Path, predictions_out: Path | None = None) -> dict:
    dev_a, dev_b, test, labels, info = load_fd004(root, partitions)
    train = build_examples(dev_a)
    val = build_examples(dev_b)
    features = [c for c in train.columns if c not in RESERVED]
    model = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, l2_regularization=2.0,
                                           random_state=42, early_stopping=False)
    model.fit(train[features].to_numpy(), train.rul.to_numpy() <= 30)
    vp = model.predict_proba(val[features].to_numpy())[:, 1]
    vy = val.rul.to_numpy() <= 30
    scored = [(t, confusion(vp >= t, vy)) for t in THRESHOLDS]
    threshold, val_metrics = sorted(scored, key=lambda x: (-x[1]["f1"], abs(x[0] - 0.5), x[0]))[0]
    # Official FD004 test engines are read only after the model and threshold are fixed.
    ex = build_examples(test.iloc[:0], test, labels)
    if len(ex) != info["eligible_test_engines"]:
        raise ValueError("Unexpected FD004 test population")
    tp = model.predict_proba(ex[features].to_numpy())[:, 1]
    ty = ex.rul.to_numpy() <= 30
    report = {
        "model": "Classical HistGradientBoostingClassifier, not OpenTime",
        "purpose": "like-for-like baseline for the OpenTime FD004 run (same FD004 split and normalisation)",
        "source_md5": SOURCE_MD5, "subset_info": info, "feature_count": len(features),
        "train_windows": len(train), "validation_windows": len(val),
        "threshold_from_validation": threshold, "validation": val_metrics,
        "official_fd004_test": confusion(tp >= threshold, ty),
        "test_labels_used_for_training_or_threshold": False,
    }
    if predictions_out is not None:  # per-engine rows stay out of the repository (gitignored artifacts/)
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        ex.assign(probability=tp, warning=tp >= threshold)[["unit", "cycle", "rul", "probability", "warning"]].to_csv(predictions_out, index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("docs/evidence/opentime-fd004/classical-same-split.json"))
    p.add_argument("--predictions-out", type=Path, default=Path("artifacts/fd004_classical_same_split_predictions.csv"))
    a = p.parse_args()
    download_dataset(a.data_dir)
    r = run(a.data_dir, a.output, a.predictions_out)
    print(json.dumps({"threshold": r["threshold_from_validation"], "validation_f1": r["validation"]["f1"],
                      "fd004_test": r["official_fd004_test"]}, indent=2))


if __name__ == "__main__":
    main()
