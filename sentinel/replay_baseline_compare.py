"""Fair, frozen same-window classical-vs-compact FD001 warning comparison.

Reproduce the original gradient-boosting baseline at 100 held-out endpoints
before applying that unchanged trained model to the frozen 2163 compact-model
history windows. No test-set tuning, paid GPU, or OpenTime 5B claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import download_dataset, load_fd001
from .features import build_examples, split_engine_ids, window_features

BASELINE_PREDICTION_SHA256 = "eb6ba0cbc4007ea32c8da1ce27b8e4818ae3a53f217129975aa9c638bf5cccd3"
FROZEN_COMPACT_REPLAY_SHA256 = "5c861bb149562571d3fefe72541ab38355596816184816c8a8b72c88d944114d"
HORIZON = 30
WINDOW = 30


def _check(path: Path, digest: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen evidence checksum mismatch: {path.name}")


def scoring(frame: pd.DataFrame, *, expected_engines=100) -> dict:
    required={"unit","cycle","actual_rul","predicted_rul","warning"}
    if frame.empty or required.difference(frame.columns):
        raise ValueError("Missing model scoring data")
    if frame.duplicated(["unit","cycle"]).any() or frame.unit.nunique()!=expected_engines:
        raise ValueError("Duplicate windows or missing engines")
    for k in ["unit","cycle","actual_rul","predicted_rul"]:
        if not np.isfinite(pd.to_numeric(frame[k]).to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite scoring values")
    if frame["warning"].dtype!=bool:
        raise ValueError("Expected strict boolean warning predictions")
    y=(frame.actual_rul<=HORIZON).to_numpy(dtype=bool)
    w=frame.warning.to_numpy(dtype=bool)
    tp,fp,fn,tn=[int(x) for x in [(y&w).sum(),(~y&w).sum(),(y&~w).sum(),(~y&~w).sum()]]
    eligible=detected=early=0
    for unit,g in frame.groupby("unit",sort=True):
        inside=g.actual_rul<=HORIZON
        alarm=g.warning
        if inside.any():
            eligible+=1
            if (inside&alarm).any(): detected+=1
        if ((~inside)&alarm).any(): early+=1
    return {
        "window_confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "window_warning_f1":float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        "window_rul_mae":float(np.abs(frame.actual_rul-frame.predicted_rul).mean()),
        "engines":{"total":expected_engines,"eligible":eligible,"detected":detected,
                   "premature_alert":early,"never_observed_within_horizon":expected_engines-eligible},
    }


def join_compact_and_classical(compact: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    for f in (compact,baseline):
        if f.duplicated(["unit","cycle"]).any():
            raise ValueError("duplicate engine/cycle in paired predictions")
    if not {"unit","cycle","actual_rul","predicted_rul","warning_probability"}.issubset(compact.columns):
        raise ValueError("Compact source lacks required fields")
    if not {"unit","cycle","actual_rul","predicted_rul","warning"}.issubset(baseline.columns):
        raise ValueError("Classical source lacks required fields")
    x=compact.merge(baseline,on=["unit","cycle"],how="inner",validate="one_to_one",suffixes=("_compact","_baseline"))
    if len(x)!=len(compact) or len(x)!=len(baseline):
        raise ValueError("engine/cycle coverage differs across models")
    if not np.array_equal(x.actual_rul_compact.to_numpy(dtype=float),x.actual_rul_baseline.to_numpy(dtype=float)):
        raise ValueError("Ground truth labels differ between paired models")
    x["warning_compact"]=(x.warning_probability>=.5).astype(bool)
    return x.sort_values(["unit","cycle"]).reset_index(drop=True)


def fit_reproduced_baseline(train,test,offsets,pinned_reference):
    examples=build_examples(train,test,offsets)
    dev=examples.loc[examples.split_source=="development"]
    holdout=examples.loc[examples.split_source=="official_test"]
    ids,_=split_engine_ids(dev)
    features=[c for c in examples.columns if c not in {"split_source","unit","cycle","rul","failure_30"}]
    model=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,
                                         l2_regularization=2.0,random_state=42,
                                         early_stopping=False)
    subset=dev.loc[dev.unit.isin(ids)]
    model.fit(subset[features].to_numpy(),subset.rul.to_numpy())
    predicted=np.maximum(0.,model.predict(holdout[features].to_numpy()))
    ref=pinned_reference.sort_values("unit").reset_index(drop=True)
    expected=holdout.sort_values("unit").reset_index(drop=True)
    if len(ref)!=100 or len(expected)!=100:
        raise ValueError("Baseline endpoint count not 100")
    if not np.array_equal(ref[["unit","cycle","rul"]].to_numpy(),expected[["unit","cycle","rul"]].to_numpy()):
        raise ValueError("Original baseline endpoint truth or cycles do not reproduce")
    if not np.allclose(ref.predicted_rul.to_numpy(),predicted,atol=1e-4,rtol=0):
        raise ValueError("Baseline final-endpoint model predictions do not reproduce")
    return model,features


def predict_past_only(test,compact,model,columns):
    engines={int(u):g for u,g in test.groupby("unit",sort=True)}
    inputs=[]
    for row in compact.itertuples(index=False):
        unit=int(row.unit);cycle=int(row.cycle)
        if unit not in engines or cycle<WINDOW or cycle>len(engines[unit]):
            raise ValueError("Invalid longitudinal engine/cycle window")
        group=engines[unit].iloc[cycle-WINDOW:cycle]
        if not np.array_equal(group.cycle.to_numpy(),np.arange(cycle-WINDOW+1,cycle+1)):
            raise ValueError("Noncontiguous past-only history")
        inputs.append(window_features(group,cycle))
    raw=pd.DataFrame(inputs)[columns]
    pred=np.maximum(0.,model.predict(raw.to_numpy()))
    return pd.DataFrame({
        "unit":compact.unit.to_numpy(),"cycle":compact.cycle.to_numpy(),
        "actual_rul":compact.actual_rul.to_numpy(),
        "predicted_rul":pred,"warning":pred<=HORIZON,
    })


def run(data_dir: Path, artifacts: Path, output: Path, *, download=False) -> dict:
    _check(artifacts/"official_test_predictions.csv",BASELINE_PREDICTION_SHA256)
    _check(artifacts/"longitudinal_replay_full.csv",FROZEN_COMPACT_REPLAY_SHA256)
    if download:
        download_dataset(data_dir)
    train,test,offsets=load_fd001(data_dir)
    compact=pd.read_csv(artifacts/"longitudinal_replay_full.csv")
    reference=pd.read_csv(artifacts/"official_test_predictions.csv")
    model,cols=fit_reproduced_baseline(train,test,offsets,reference)
    baseline=predict_past_only(test,compact,model,cols)
    paired=join_compact_and_classical(compact,baseline)
    compact_rows=pd.DataFrame({"unit":paired.unit,"cycle":paired.cycle,
                               "actual_rul":paired.actual_rul_compact,
                               "predicted_rul":paired.predicted_rul_compact,
                               "warning":paired.warning_compact})
    if len(paired)!=2163:
        raise ValueError("Frozen longitudinal replay count changed")
    samewindow={"compact":scoring(compact_rows),"classical":scoring(baseline)}
    differences=dict(compact_only_risk_detection=0,baseline_only_risk_detection=0,
                     both_detect_risk=0,neither_detect_risk=0,
                     compact_only_premature_alert=0,baseline_only_premature_alert=0)
    for unit,g in paired.groupby("unit"):
        inside=g.actual_rul_compact<=HORIZON
        c,b=g.warning_compact,g.warning
        if inside.any():
            c_hit=bool((inside&c).any());b_hit=bool((inside&b).any())
            label=("both_detect_risk" if c_hit and b_hit else
                   "compact_only_risk_detection" if c_hit else
                   "baseline_only_risk_detection" if b_hit else "neither_detect_risk")
            differences[label]+=1
        c_early=bool(((~inside)&c).any());b_early=bool(((~inside)&b).any())
        if c_early and not b_early:differences["compact_only_premature_alert"]+=1
        if b_early and not c_early:differences["baseline_only_premature_alert"]+=1
    report={
        "study":"Same-window FD001 longitudinal warning comparison",
        "dataset":"100 NASA simulated held-out engines, 2163 five-cycle-sampled windows",
        "compact_model":"Original independent compact temporal QA warning head (NOT OpenTime 5B)",
        "classical_model":"Original HistGradientBoostingRegressor reproduced from development engines only",
        "baseline_original_100_endpoints_reproduced":True,
        "compact_original_100_endpoints_reproduced_by_source_run":True,
        "same_windows_and_labels":True,
        "thresholds":"Compact p>=0.5; classical predicted RUL<=30; no test tuning",
        **samewindow,"paired_engine_differences":differences,
        "original_baseline_predictions_sha256":BASELINE_PREDICTION_SHA256,
        "original_compact_replay_predictions_sha256":FROZEN_COMPACT_REPLAY_SHA256,
        "limitations":[
            "Correlated windows are not 2163 independent engines.",
            "75 test engines lack observations inside the final 30-cycle risk horizon.",
            "Both evaluated models are independent CPU models, NOT the organizer OpenTime 5B.",
            "Retrospective simulated FD001 histories, not real maintenance savings or safety validation.",
        ],
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"paired_replay_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    p.add_argument("--artifact-dir",type=Path,default=Path("artifacts/frozen_replay"))
    p.add_argument("--output-dir",type=Path,default=Path("artifacts/paired_replay"))
    p.add_argument("--download",action="store_true")
    a=p.parse_args()
    print(json.dumps(run(a.data_dir,a.artifact_dir,a.output_dir,download=a.download),indent=2))

if __name__=="__main__":
    main()
