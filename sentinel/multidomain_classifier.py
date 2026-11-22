"""Exploratory FD004 holdout: development-selected multi-domain warning classifier.

A classical CPU classifier is trained on FD001/FD002/FD003 development engines,
with its warning threshold chosen *only* on those cohorts' disjoint validation
engines. FD004 remains completely excluded until the final once-only evaluation.
This is post-hoc exploratory work after the previous domain-shift audit, NOT an
OpenTime TSLM result, a real-aircraft safety estimate or a preregistered study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import SOURCE_MD5, SOURCE_URL, download_dataset
from .features import build_examples
from .multidomain_audit import TRAIN_COHORTS, TARGET_COHORT, RESERVED, _load_training_source, _split_one_cohort
from .transfer_audit import _load_test_cohort

THRESHOLDS = tuple(float(x) for x in np.linspace(0.1, 0.9, 17))


def warning_scores(truth: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    truth = np.asarray(truth, dtype=bool)
    probs = np.asarray(probs, dtype=float)
    if truth.shape != probs.shape or not np.isfinite(probs).all() or ((probs < 0)|(probs > 1)).any():
        raise ValueError("Bad warning probabilities or truth dimensions")
    positive = probs >= threshold
    tp = int(np.count_nonzero(truth & positive))
    fp = int(np.count_nonzero(~truth & positive))
    fn = int(np.count_nonzero(truth & ~positive))
    tn = int(np.count_nonzero(~truth & ~positive))
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":p,"recall":r,"f1":f1,
            "brier":float(np.mean((probs-truth.astype(float))**2)),"count":len(truth)}


def choose_validation_threshold(truth: np.ndarray, probs: np.ndarray) -> tuple[float,dict]:
    """Max validation F1 on a *predefined* grid, tie toward default 0.5."""
    choices=[(threshold,warning_scores(truth,probs,threshold)) for threshold in THRESHOLDS]
    best=sorted(choices,key=lambda item:(-item[1]["f1"],abs(item[0]-0.5),item[0]))[0]
    return best[0],best[1]


def run(data_dir: Path,output_dir: Path,*,download=False) -> dict:
    if download:download_dataset(data_dir)
    archive_path=data_dir/"CMAPSSData.zip"
    if not archive_path.is_file() or hashlib.md5(archive_path.read_bytes()).hexdigest()!=SOURCE_MD5:
        raise ValueError("Source missing or MD5 unverified")
    examples_train,examples_val,cohort_counts=[],[],[]
    features=None
    with ZipFile(archive_path) as z:
        for cohort in TRAIN_COHORTS:
            raw=_load_training_source(z,cohort)
            train,val,count=_split_one_cohort(cohort,raw)
            available=[c for c in train.columns if c not in RESERVED]
            if features is None:features=available
            if features!=available:raise ValueError("Cross-cohort feature mismatch")
            examples_train.append(train)
            examples_val.append(val)
            cohort_counts.append(count)
        train=pd.concat(examples_train,ignore_index=True)
        val=pd.concat(examples_val,ignore_index=True)
        classifier=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,
                                      l2_regularization=2.0,random_state=42,early_stopping=False)
        classifier.fit(train[features].to_numpy(),(train.rul.to_numpy()<=30))
        val_probability=classifier.predict_proba(val[features].to_numpy())[:,1]
        threshold, val_metrics=choose_validation_threshold(val.rul.to_numpy()<=30,val_probability)
        # Target FD004 is read only AFTER training AND threshold lock.
        target,labels=_load_test_cohort(z,TARGET_COHORT)
        eligible=set(int(u) for u in target.groupby('unit').size().loc[lambda v:v>=30].index)
        examples=build_examples(target.iloc[:0],target,labels)
        if set(int(u) for u in examples.unit.unique())!=eligible or not eligible:
            raise ValueError("Unexpected official FD004 endpoint population")
        test_prob=classifier.predict_proba(examples[features].to_numpy())[:,1]
        test_metrics=warning_scores(examples.rul.to_numpy()<=30,test_prob,threshold)
        test_metrics.update({"eligible_engines":len(examples),
                            "total_official_test_engines":int(target.unit.nunique()),
                            "excluded_due_to_short_history":int(target.unit.nunique())-len(examples),
                            "warning_prevalence":float(np.mean(examples.rul.to_numpy()<=30))})
    report={
        "project":"MUSITU AXIOM Sentinel",
        "experiment":"post-hoc exploratory multi-domain direct-warning classifier held out FD004",
        "model":"Classical HistGradientBoostingClassifier, not temporal QA or OpenTime 5B",
        "source":SOURCE_URL,"source_md5":SOURCE_MD5,
        "training_only_cohorts":list(TRAIN_COHORTS),"heldout_test_cohort":TARGET_COHORT,
        "training_counts":cohort_counts,
        "feature_count":len(features),
        "warning_horizon_cycles":30,
        "target_fd004_data_used_to_fit_or_select_threshold":False,
        "validation_threshold_candidates":list(THRESHOLDS),
        "chosen_validation_threshold":threshold,
        "independent_development_validation":val_metrics,
        "official_fd004_test":test_metrics,
        "timestamp":datetime.now(timezone.utc).isoformat(),
        "limitations":[
            "This experiment was devised post-hoc after inspecting frozen FD001-only transfer failures; it is exploratory, not preregistered.",
            "A classical CPU warning classifier trained with three NASA simulated development subsets, not the organizer's OpenTime TSLM.",
            "FD004 data was strictly excluded from fitting and threshold selection, but another simulation population is not a real-world manufacturer shift.",
            "Selection used the validation cohort engines and a fixed 0.1-0.9 probability grid; no official FD004 test label tuning was performed.",
            "Short-history official FD004 engines (<30 cycles) were excluded and counted; no reliable claims for them.",
            "No source NASA rows, individual engine labels, or trained model weights redistributed.",
            "No financial savings, real aircraft safety or real-world operation inferred.",
        ]
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'multidomain_warning_aggregate.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"validation":{"threshold":threshold,"f1":round(val_metrics['f1'],4),
                                      "precision":round(val_metrics['precision'],4),"recall":round(val_metrics['recall'],4)},
                      "fd004":{"f1":round(test_metrics['f1'],4),"precision":round(test_metrics['precision'],4),
                                "recall":round(test_metrics['recall'],4),"tp":test_metrics['tp'],
                                "fp":test_metrics['fp'],"fn":test_metrics['fn'],"tn":test_metrics['tn']}},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/multidomain-warning'))
    p.add_argument('--download',action='store_true')
    a=p.parse_args();run(a.data_dir,a.output_dir,download=a.download)

if __name__=='__main__':main()
