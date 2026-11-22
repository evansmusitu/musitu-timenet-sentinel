"""Frozen-model sequence-ablation protocol contracts (no GPU or dataset download)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.sequence_integrity import perturb, verify_frozen_reproduction, score_condition


def _windows():
    base = np.arange(4 * 24, dtype=np.float32).reshape(4, 24)
    return np.stack([base + i * 100 for i in range(3)])


def test_last_value_only_retains_current_observation_but_removes_history():
    original = _windows()
    altered = perturb(original, "last_value_only", seed=7)
    assert altered.shape == original.shape
    np.testing.assert_array_equal(altered[:, -1, :], original[:, -1, :])
    np.testing.assert_array_equal(altered[:, 0, :], original[:, -1, :])
    np.testing.assert_array_equal(original[0, 0], np.arange(24, dtype=np.float32))


def test_reverse_preserves_each_channel_values_but_reverses_time_order():
    original = _windows()
    altered = perturb(original, "reverse_time", seed=7)
    np.testing.assert_array_equal(altered[:, 0], original[:, -1])
    np.testing.assert_array_equal(altered[:, -1], original[:, 0])


def test_shuffle_uses_same_temporal_permutation_across_all_channels_and_is_deterministic():
    original = _windows()
    original_copy = original.copy()
    altered = perturb(original, "shuffle_time", seed=42)
    np.testing.assert_array_equal(altered, perturb(original, "shuffle_time", seed=42))
    assert not np.array_equal(altered[0], original[0])
    for engine in range(3):
        for row in altered[engine]:
            matching = np.flatnonzero((original[engine] == row).all(axis=1))
            assert len(matching) == 1
    np.testing.assert_array_equal(original, original_copy)


def test_mean_value_control_preserves_channel_means():
    original = _windows()
    altered = perturb(original, "mean_value_only", seed=1)
    np.testing.assert_allclose(altered.mean(axis=1), original.mean(axis=1))
    np.testing.assert_allclose(altered[:, 0], altered[:, -1])


@pytest.mark.parametrize("invalid", ["not-a-mode", "", "random_mask"])
def test_unknown_condition_fails_closed(invalid):
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb(_windows(), invalid)


def test_rejects_nonfinite_and_bad_telemetry_dimensions():
    with pytest.raises(ValueError, match="24"):
        perturb(np.zeros((1, 30, 23), dtype=np.float32), "reverse_time")
    with pytest.raises(ValueError, match="finite"):
        perturb(np.full((1, 30, 24), np.nan), "reverse_time")


def test_verified_reproduction_requires_all_engines_and_same_frozen_warnings():
    ref = pd.DataFrame({"unit":[1,2],"observed_cycle":[20,50],"actual_rul":[10,80],
                        "predicted_rul":[12.5,70.0],"risk_within_30_probability":[.8,.2],
                        "risk_within_30_predicted":[True,False]})
    meta = [(1,20),(2,50)]
    verify_frozen_reproduction(ref, meta, np.array([10,80]), np.array([12.5,70]),
                               np.array([True,False]), np.array([.8,.2]))
    with pytest.raises(ValueError, match="reference"):
        verify_frozen_reproduction(ref, meta, np.array([10,80]), np.array([12.5,70]),
                                   np.array([False,False]), np.array([.8,.2]))
    with pytest.raises(ValueError, match="engine"):
        verify_frozen_reproduction(ref, [(2,50),(1,20)], np.array([80,10]),
                                   np.array([70,12.5]), np.array([False,True]),np.array([.2,.8]))


def test_metrics_evaluate_original_threshold_without_optimizing_on_test_labels():
    s = score_condition(np.array([5., 45., 22., 120.]),np.array([10.,50.,80.,110.]),
                        np.array([True,False,False,False]), np.array([.8,.3,.1,.2]))
    assert (s["tp"],s["fp"],s["fn"],s["tn"]) == (1,0,1,2)
    assert s["n"] == 4
    assert s["warning_f1"] == pytest.approx(2/3)
    assert s["rul_mae_cycles"] == pytest.approx((5+5+58+10)/4)
