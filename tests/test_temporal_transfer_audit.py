"""Compact checkpoint identity and test-only evaluation invariants."""
import numpy as np
import pytest
from pathlib import Path
from zipfile import ZipFile
from sentinel.transfer_audit import COHORTS, _load_test_cohort
from sentinel.temporal_transfer_audit import _evaluate, _load_trusted_original_checkpoint
from sentinel.temporal_qa import build_model
from tests.test_transfer_audit import _toy_source

torch=pytest.importorskip('torch')


def test_safe_own_checkpoint_round_trip(tmp_path):
    model=build_model()
    checkpoint={"model_state": model.state_dict(),
                "mean":np.zeros((1,1,24),dtype=np.float32),
                "scale":np.ones((1,1,24),dtype=np.float32),
                "questions":{"warning_30":"Could this engine fail within 30 cycles?"},
                "answer_tokens":"test"}
    path=tmp_path/"safe.pt";torch.save(checkpoint,path)
    loaded,norm=_load_trusted_original_checkpoint(path)
    assert len(list(loaded.parameters()))>0
    assert norm[0].shape==(1,1,24)


def test_reject_invalid_checkpoint_schema(tmp_path):
    path=tmp_path/"bad.pt";torch.save({"bad":1},path)
    with pytest.raises(ValueError,match="unexpected"): _load_trusted_original_checkpoint(path)


def test_reject_bad_checkpoint_normalization(tmp_path):
    model=build_model()
    path=tmp_path/"bad.pt"
    checkpoint={"model_state":model.state_dict(),"mean":np.zeros((1,1,24),dtype=np.float32),
                "scale":np.zeros((1,1,24),dtype=np.float32),"questions":{},"answer_tokens":"test"}
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match="scale"): _load_trusted_original_checkpoint(path)


def test_synthetic_unseen_engine_predictions_only(tmp_path):
    path=tmp_path/"toy.zip";_toy_source(path)
    with ZipFile(path) as archive: test,labels=_load_test_cohort(archive,"FD003")
    model=build_model()
    norm=(np.zeros((1,1,24),dtype=np.float32),np.ones((1,1,24),dtype=np.float32))
    result=_evaluate(model,norm,test,labels)
    assert result["heldout_test_engines"]==2
    assert sum(result["confusion"].values())==2
    assert 0<=result["warning_probability_brier"]<=1


def test_no_false_cross_vendor_claim_in_supported_cohorts():
    assert COHORTS=={"FD001":(1,1),"FD002":(6,1),"FD003":(1,2),"FD004":(6,2)}
