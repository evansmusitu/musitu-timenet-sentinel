"""Synthetic leakage and target-exclusion tests for exploratory cross-domain experiment."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from sentinel.multidomain_audit import TRAIN_COHORTS, TARGET_COHORT, _load_training_source, _split_one_cohort, _evaluate
from sentinel.data import COLS


def _fixture(archive: Path):
    rows=[]
    for unit in (1,2,3,4,5,6,7,8,9,10):
        for cycle in range(1,36):
            vals=[unit,cycle]+[float(unit+cycle/100)]*(len(COLS)-2)
            rows.append(" ".join(map(str, vals))+"\n")
    with ZipFile(archive,"w") as f:
        for cohort in TRAIN_COHORTS:
            f.writestr(f"train_{cohort}.txt","".join(rows))


def test_source_loader_excludes_target_subset(tmp_path):
    archive=tmp_path/"toy.zip";_fixture(archive)
    with ZipFile(archive) as z:
        for cohort in TRAIN_COHORTS:
            frame=_load_training_source(z,cohort)
            assert len(frame)==10*35
        with pytest.raises(ValueError,match="forbidden"):
            _load_training_source(z,TARGET_COHORT)


def test_engine_group_isolation_with_overlapping_numeric_ids_between_sources(tmp_path):
    archive=tmp_path/"toy.zip";_fixture(archive)
    with ZipFile(archive) as z:
        counts=[]
        for cohort in TRAIN_COHORTS:
            frame=_load_training_source(z,cohort)
            train,val,c=_split_one_cohort(cohort,frame)
            assert set(train.unit.unique()).isdisjoint(set(val.unit.unique()))
            assert c["train_engines"]+c["validation_engines"]==10
            counts.append(c)
        assert len({c["cohort"] for c in counts})==3
        assert all(c["train_windows"] and c["validation_windows"] for c in counts)


def test_holdout_eval_excludes_short_history_and_never_trains(tmp_path):
    rows=[]
    for unit,cycles in [(1,31),(2,20),(3,31)]:
        for cycle in range(1,cycles+1):
            rows.append([unit,cycle]+[float(cycle)]*(len(COLS)-2))
    frame=pd.DataFrame(rows,columns=COLS)
    from sentinel.features import build_examples
    example=build_examples(frame.iloc[:0],frame,{1:5,2:30,3:40})
    features=[col for col in example.columns if col not in {"split_source","unit","cycle","rul","failure_30"}]
    class Stub:
        def predict(self,x):
            assert x.shape[0]==2
            return np.array([10.,60.])
    result=_evaluate(Stub(),frame,{1:5,2:30,3:40},features)
    assert result["excluded_due_to_fewer_than_30_observations"]==1
    assert result["heldout_test_engines"]==2
    assert result["confusion"]=={"tp":1,"fp":0,"fn":0,"tn":1}
