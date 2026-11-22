"""CPU classifier gate keeps FD004 official test out of validation threshold selection."""
import numpy as np
import pytest
from sentinel.multidomain_classifier import warning_scores,choose_validation_threshold,THRESHOLDS


def test_threshold_grid_is_fixed_before_test():
    assert THRESHOLDS[0]==pytest.approx(0.1)
    assert THRESHOLDS[-1]==pytest.approx(0.9)
    assert len(THRESHOLDS)==17
    assert 0.5 in THRESHOLDS


def test_binary_scores_use_correct_confusion():
    x=warning_scores(np.array([1,1,0,0],dtype=bool),np.array([0.9,0.1,0.8,0.2]),0.5)
    assert (x['tp'],x['fp'],x['fn'],x['tn'])==(1,1,1,1)
    assert x['f1']==pytest.approx(0.5)


def test_validation_threshold_tie_prefers_half():
    t,p=choose_validation_threshold(np.array([True,False]),np.array([0.99,0.01]))
    assert t==pytest.approx(0.5)
    assert p['f1']==1.0

@pytest.mark.parametrize('probs',[[-.1,1.],[1.01,0.],[float('nan'),0.]])
def test_reject_invalid_model_probabilities(probs):
    with pytest.raises(ValueError):warning_scores(np.array([True,False]),np.asarray(probs),0.5)
