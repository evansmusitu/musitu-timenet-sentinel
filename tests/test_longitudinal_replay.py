"""Synthetic-only tests of the frozen longitudinal maintenance replay."""
import numpy as np
import pandas as pd
import pytest

from sentinel.data import COLS
from sentinel.longitudinal_replay import make_history, summarize_replay


def engine(unit,n):
    return pd.DataFrame([[unit,c,*[float(unit*10+c+i/1000) for i in range(24)]]
                         for c in range(1,n+1)],columns=COLS)


def test_past_only_exact_labels_and_final_endpoint():
    raw=pd.concat([engine(1,35),engine(2,42)],ignore_index=True)
    x,rows=make_history(raw,{1:5,2:45},stride=5)
    assert rows[["unit","cycle","actual_rul"]].values.tolist()==[
        [1,30,10.],[1,35,5.],[2,30,57.],[2,35,52.],[2,40,47.],[2,42,45.]]
    assert x.shape==(6,30,24)
    np.testing.assert_allclose(x[0,-1,:],engine(1,35).iloc[29][COLS[2:]].to_numpy(dtype=float))


def test_future_sensor_change_never_changes_past_window():
    raw=engine(3,36)
    x,rows=make_history(raw,{3:0},stride=5)
    assert rows.cycle.tolist()==[30,35,36]
    assert rows.actual_rul.tolist()==[6,1,0]
    raw.loc[35,COLS[2]]=99999.
    later,_=make_history(raw,{3:0},stride=5)
    np.testing.assert_array_equal(x[0],later[0])
    assert not np.array_equal(x[-1],later[-1])


def test_invalid_source_fails_closed():
    with pytest.raises(ValueError,match="missing"):
        make_history(engine(1,35),{},stride=5)
    with pytest.raises(ValueError,match="nonnegative"):
        make_history(engine(1,35),{1:-3},stride=5)
    with pytest.raises(ValueError,match="integral"):
        make_history(engine(1,35),{1:2.5},stride=5)
    with pytest.raises(ValueError,match="short"):
        make_history(engine(1,20),{1:0},stride=5)
    with pytest.raises(ValueError,match="stride"):
        make_history(engine(1,35),{1:0},stride=0)
    damaged=engine(1,35)
    damaged.loc[30,"cycle"]=999
    with pytest.raises(ValueError,match="contiguous"):
        make_history(damaged,{1:0},stride=5)


def test_aggregation_distinguishes_early_false_alerts_from_valid_risk():
    frame=pd.DataFrame({
        "unit":[1,1,1,2,2,3,3],
        "cycle":[30,35,40,30,35,30,35],
        "actual_rul":[40,35,30,60,55,10,5],
        "warning_probability":[.1,.9,.9,.1,.2,.1,.2],
        "predicted_rul":[35,25,25,75,65,20,10],
    })
    result=summarize_replay(frame,expected_engines=3)
    assert result["window_confusion"]=={"tp":1,"fp":1,"fn":2,"tn":3}
    assert result["engines"]["with_in_horizon_observations"]==2
    assert result["engines"]["with_in_horizon_warning"]==1
    assert result["engines"]["with_out_of_horizon_false_alert"]==1
    assert result["engines"]["without_observed_in_horizon_data"]==1


def test_corrupt_predictions_rejected():
    good=pd.DataFrame({"unit":[1,1],"cycle":[30,35],"actual_rul":[5,0],
                       "warning_probability":[.4,.7],"predicted_rul":[10.,1.]})
    for bad in (pd.concat([good,good.iloc[[1]]]),
                good.assign(warning_probability=[1.1,.7]),
                good.assign(predicted_rul=[np.nan,1.])):
        with pytest.raises(ValueError):
            summarize_replay(bad,expected_engines=1)
