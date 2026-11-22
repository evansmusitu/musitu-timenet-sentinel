"""Validate failure-closed NASA transfer experiment without source data or GPU."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from sentinel.transfer_audit import _archive_members, _load_test_cohort, _evaluate
from sentinel.data import COLS


def _toy_source(zip_path: Path, suffix="FD003", labels="11\n23\n", duplicate=False):
    lines=[]
    for unit in (1,2):
        for cycle in range(1,31):
            row=[unit, cycle] + [1.0]*(len(COLS)-2)
            lines.append(" ".join(map(str,row))+"\n")
    with ZipFile(zip_path,"w") as f:
        f.writestr(f"test_{suffix}.txt", "".join(lines))
        if duplicate:
            f.writestr(f"other/test_{suffix}.txt", "".join(lines))
        f.writestr(f"RUL_{suffix}.txt",labels)


def test_synthetic_test_only_parses_two_engines(tmp_path):
    archive=tmp_path/"toy.zip";_toy_source(archive)
    with ZipFile(archive) as z: frame, labels=_load_test_cohort(z,"FD003")
    assert labels=={1:11,2:23}
    assert sorted(frame.unit.unique().tolist())==[1,2]
    assert len(frame)==60
    assert frame.cycle.max()==30

@pytest.mark.parametrize("labels",["11\n","11\n23\n99\n","-1\n23\n","NaN\n23\n","11.5\n23\n"])
def test_malformed_official_label_rejected(tmp_path, labels):
    archive=tmp_path/"toy.zip";_toy_source(archive,labels=labels)
    with ZipFile(archive) as z,pytest.raises(ValueError): _load_test_cohort(z,"FD003")


def test_reject_ambiguous_zip_duplicate_filename(tmp_path):
    archive=tmp_path/"toy.zip";_toy_source(archive,duplicate=True)
    with ZipFile(archive) as z,pytest.raises(ValueError,match="ambiguous"):
        _load_test_cohort(z,"FD003")


def test_reject_unavailable_cohort(tmp_path):
    archive=tmp_path/"toy.zip";_toy_source(archive)
    with ZipFile(archive) as z,pytest.raises(ValueError,match="Unsupported"):
        _load_test_cohort(z,"FD005")


def test_model_never_receives_labels_as_features(tmp_path):
    archive=tmp_path/"toy.zip";_toy_source(archive)
    with ZipFile(archive) as z: frame,labels=_load_test_cohort(z,"FD003")
    class CapturePredict:
        def predict(self,x):
            assert x.shape==(2,1)
            assert np.array_equal(x[:,0],[30,30]);return np.array([10,35])
    # Simulate a one-feature model; actual labels are still used only for scoring.
    from sentinel.features import build_examples
    fake_train=frame.iloc[:0]
    example=build_examples(fake_train,frame,labels)
    # feature column cycle is not available to the real benchmark feature contract;
    # use age_cycles (an allowed past-only feature) in this miniature test.
    results=_evaluate(CapturePredict(),["age_cycles"],fake_train,frame,labels)
    assert results["confusion"]=={"tp":1,"fp":0,"fn":1,"tn":0}
    assert results["heldout_test_engines"]==2
