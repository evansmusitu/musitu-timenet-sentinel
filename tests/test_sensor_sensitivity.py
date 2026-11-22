"""Synthetic-only contracts for noncausal frozen-model sensor sensitivity."""
import numpy as np
import pytest
from sentinel.sensor_sensitivity import ablate_channels, select_case_observation, summarized_channels


def sample_data():
    x=np.ones((2,3,24),dtype=np.float32)
    x[0,:,0]=[.2,.4,.6]
    x[1,:,0]=[.6,.8,1.0]
    return x


def fake_risk(batch):
    return batch[:,:,0].mean(axis=1)


def test_mean_replacement_is_nonmutating_and_measures_correct_direction():
    x=sample_data();old=x.copy()
    mean=np.full((1,1,24),.5,dtype=np.float32)
    base=fake_risk(x)
    matrix=ablate_channels(x,mean,base,fake_risk)
    assert matrix.shape==(24,2)
    np.testing.assert_allclose(matrix[0],[.5,.5],atol=1e-6)
    np.testing.assert_allclose(matrix[1],base)
    np.testing.assert_array_equal(x,old)
    info=summarized_channels(matrix,base,['c'+str(i) for i in range(24)])
    assert info[0]['channel']=='c0'
    assert info[0]['mean_absolute_probability_change_percentage_points']==pytest.approx(20.,abs=1e-4)


def test_malformed_source_and_perturbed_outputs_fail_closed():
    x=sample_data();m=np.zeros((1,1,24),dtype=np.float32);p=fake_risk(x)
    with pytest.raises(ValueError,match="24"):
        ablate_channels(x[:,:,:23],m,p,fake_risk)
    with pytest.raises(ValueError,match="mean"):
        ablate_channels(x,np.zeros((24,)),p,fake_risk)
    with pytest.raises(ValueError,match="baseline"):
        ablate_channels(x,m,np.array([.5]),fake_risk)
    with pytest.raises(ValueError,match="finite"):
        ablate_channels(x,m,np.array([np.nan,.2]),fake_risk)
    with pytest.raises(ValueError,match="prediction"):
        ablate_channels(x,m,p,lambda xx:np.array([.3]))
    with pytest.raises(ValueError,match="probability"):
        ablate_channels(x,m,p,lambda xx:np.full(len(xx),-0.3))


def case(category,ys,ps):
    return {'engine_id':3,'category':category,'sampled_history':[
       {'cycle':30+5*i,'actual_rul':y,'warning_probability':p,'predicted_rul':42.}
       for i,(y,p) in enumerate(zip(ys,ps))]}


def test_each_declared_case_reproduces_its_actual_outcome():
    x=select_case_observation(case('caught_risk',[45,30,25],[.2,.7,.8]))
    assert x['cycle']==35 and x['actual_rul']==30
    x=select_case_observation(case('missed_risk',[40,25,20],[.1,.4,.3]))
    assert x['cycle']==40
    x=select_case_observation(case('premature_alert',[70,50,25],[.3,.8,.9]))
    assert x['cycle']==35 and x['actual_rul']==50
    x=select_case_observation(case('clean_monitoring',[60,55,50],[.2,.4,.3]))
    assert x['cycle']==40


def test_misclassified_or_reordered_case_is_rejected():
    with pytest.raises(ValueError,match="category"):
        select_case_observation(case('invalid',[50],[.2]))
    with pytest.raises(ValueError,match="category"):
        select_case_observation(case('caught_risk',[50,45],[.2,.3]))
    x=case('clean_monitoring',[50,45],[.2,.3])
    x['sampled_history'][1]['cycle']=30
    with pytest.raises(ValueError,match="cycle"):
        select_case_observation(x)
