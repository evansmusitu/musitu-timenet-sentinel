import numpy as np
import pandas as pd
import pytest
from sentinel.data import COLS
from sentinel.features import build_examples, split_engine_ids
from sentinel.benchmark import metrics

def engines(units=(1,2,3,4), length=80):
    rows=[]
    for unit in units:
        for cycle in range(1,length+1):
            rows.append({"unit":unit,"cycle":cycle,
                **{name:float(unit+cycle)/100 for name in COLS[2:]}})
    return pd.DataFrame(rows)

def test_past_only_features():
    src=engines()
    a=build_examples(src,window=30,stride=10)
    assert a.loc[(a.unit==1)&(a.cycle==30),"rul"].iloc[0]==50
    altered=src.copy()
    altered.loc[altered.cycle>40,"sensor_1"]+=1e6
    b=build_examples(altered,window=30,stride=10)
    assert a.loc[(a.unit==1)&(a.cycle==30),"sensor_1_mean"].iloc[0] == b.loc[(b.unit==1)&(b.cycle==30),"sensor_1_mean"].iloc[0]

def test_official_test_endpoint_only():
    ex=build_examples(engines(),engines(units=(10,11),length=44),{10:17,11:22})
    test=ex[ex.split_source=="official_test"]
    assert len(test)==2
    assert sorted(test.rul.tolist())==[17,22]

def test_group_disjoint():
    df=build_examples(engines())
    tr,val=split_engine_ids(df)
    assert not set(tr).intersection(val)
    assert set(tr)|set(val)=={1,2,3,4}

def test_metrics():
    m=metrics(np.array([10.,50.]),np.array([12.,45.]))
    assert m["count"]==2
    assert m["warning_30_f1"]==1.0

def test_invalid_window():
    with pytest.raises(ValueError):
        build_examples(engines(),window=1)
