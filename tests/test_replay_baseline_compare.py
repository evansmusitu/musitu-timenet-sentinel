"""Synthetic-only contracts for fair same-window longitudinal model comparisons."""
import pandas as pd
import numpy as np
import pytest

from sentinel.replay_baseline_compare import scoring, join_compact_and_classical


def test_score_counts_engines_not_correlated_windows():
    f=pd.DataFrame({"unit":[1,1,2,2,3,3],"cycle":[30,35,30,35,30,35],
       "actual_rul":[25,20,70,65,40,35],"warning":[False,True,True,False,False,False],
       "predicted_rul":[80,8,30,50,55,43]})
    x=scoring(f,expected_engines=3)
    assert x["window_confusion"]==dict(tp=1,fp=1,fn=1,tn=3)
    assert x["engines"]["eligible"]==1 and x["engines"]["detected"]==1
    assert x["engines"]["premature_alert"]==1
    assert x["engines"]["never_observed_within_horizon"]==2


def test_join_refuses_mismatched_labels_and_duplicate_engine_cycle():
    compact=pd.DataFrame({"unit":[1,2],"cycle":[30,40],"actual_rul":[12.,42.],
       "predicted_rul":[15.,30.],"warning_probability":[.7,.2]})
    baseline=pd.DataFrame({"unit":[1,2],"cycle":[30,40],"actual_rul":[12.,42.],
       "predicted_rul":[17.,50.],"warning":[True,False]})
    z=join_compact_and_classical(compact,baseline)
    assert len(z)==2 and "warning_compact" in z
    assert z.warning_compact.tolist()==[True,False]
    with pytest.raises(ValueError,match="truth"):
        join_compact_and_classical(compact,baseline.assign(actual_rul=[12.,99.]))
    with pytest.raises(ValueError,match="duplicate"):
        join_compact_and_classical(pd.concat([compact,compact.iloc[[0]]]),baseline)
    with pytest.raises(ValueError,match="coverage"):
        join_compact_and_classical(compact,baseline.iloc[[0]])
