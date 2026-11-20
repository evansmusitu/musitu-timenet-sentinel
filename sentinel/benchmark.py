"""Reproducible grouped NASA FD001 CPU baseline, NOT OpenTSLM inference."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support
from .data import load_fd001, download_dataset, SOURCE_URL, SOURCE_MD5
from .features import build_examples, split_engine_ids

def metrics(actual, predicted):
    predicted = np.maximum(0.0, predicted)
    actual = np.asarray(actual)
    p, r, f, _ = precision_recall_fscore_support(actual <= 30, predicted <= 30, average="binary", zero_division=0)
    return {
        "count": int(len(actual)),
        "rul_rmse_cycles": float(np.sqrt(mean_squared_error(actual, predicted))),
        "rul_mae_cycles": float(mean_absolute_error(actual, predicted)),
        "warning_30_precision": float(p), "warning_30_recall": float(r),
        "warning_30_f1": float(f), "warning_30_prevalence": float(np.mean(actual<=30))
    }

def run(data_dir: Path, output_dir: Path, download=False):
    if download:
        download_dataset(data_dir)
    train, test, official_rul = load_fd001(data_dir)
    examples = build_examples(train, test, official_rul)
    dev = examples.loc[examples.split_source=="development"].copy()
    holdout = examples.loc[examples.split_source=="official_test"].copy()
    train_ids, val_ids = split_engine_ids(dev)
    train_sel = dev.unit.isin(train_ids)
    val_sel = dev.unit.isin(val_ids)
    reserved = {"split_source","unit","cycle","rul","failure_30"}
    features = [c for c in examples.columns if c not in reserved]
    model = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15,
                                          l2_regularization=2.0, random_state=42,
                                          early_stopping=False)
    model.fit(dev.loc[train_sel,features].to_numpy(),dev.loc[train_sel,"rul"].to_numpy())
    actual_val = dev.loc[val_sel,"rul"].to_numpy()
    predicted_val = model.predict(dev.loc[val_sel,features].to_numpy())
    # Official NASA test engines are untouched until this FINAL evaluation.
    actual_test = holdout.rul.to_numpy()
    predicted_test = np.maximum(0.0,model.predict(holdout[features].to_numpy()))
    report = {
        "project":"MUSITU AXIOM Sentinel", "dataset":"NASA C-MAPSS FD001",
        "source_url":SOURCE_URL, "source_md5":SOURCE_MD5,
        "evaluated_utc":datetime.now(timezone.utc).isoformat(),
        "model":"HistGradientBoostingRegressor CPU baseline",
        "tslm_trained_or_evaluated":False,
        "train_engines":len(train_ids),"validation_engines":len(val_ids),
        "official_test_engines":int(test.unit.nunique()),
        "validation":metrics(actual_val,predicted_val),
        "official_test":metrics(actual_test,predicted_test),
        "limits":"Simulation only; no real-aircraft maintenance conclusions; not a Time-Series Language Model."
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    result = holdout[["unit","cycle","rul"]].copy()
    result["predicted_rul"] = predicted_test
    result["warning_30"] = predicted_test<=30
    result.to_csv(output_dir/"official_test_predictions.csv",index=False)
    (output_dir/"summary.md").write_text(
        "# NASA FD001 baseline (not OpenTime)\n\n"
        +f"Official test MAE: {report['official_test']['rul_mae_cycles']:.3f} cycles; "
        +f"RMSE: {report['official_test']['rul_rmse_cycles']:.3f} cycles; "
        +f"30-cycle warning F1: {report['official_test']['warning_30_f1']:.3f}.\n\n"
        +"Simulated benchmark, not aircraft maintenance guidance.\n"
    )
    print(json.dumps(report,indent=2))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts"))
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,args.download)

if __name__=="__main__":
    main()
