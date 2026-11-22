"""CPU unit tests of OpenTime option likelihoods and heldout-safe calibration."""
import math
from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch", reason="Heavy OpenTime-specific CPU tests run in dedicated workflow")

from sentinel.opentime_decision import (
    check_qa_rows, choose_validation_threshold, score_option_margin,
    score_examples, summarize,
)


def test_paired_candidate_likelihood_and_unmodified_prompt():
    seen = []
    class Stub:
        device=torch.device("cpu")
        def eval(self): pass
        def loss(self, series, prompt, answer, reduction):
            assert reduction=="none"
            seen.append((prompt,answer))
            return torch.tensor([0.1,0.2] if answer=="Yes" else [0.9,0.2])
    ex=SimpleNamespace(evaluated_question=lambda:"answer_options:\n1. Yes\n2. No What?",options=["Yes","No"],series=[])
    got=score_option_margin(Stub(),ex)
    assert got == pytest.approx(0.8,abs=1e-6)
    assert [x[1] for x in seen]==["Yes","No"]
    assert seen[0][0]==seen[1][0]


def test_validation_f1_threshold_is_selected_from_validation_only():
    scores=[-2.0,-1.0,-0.3,0.5,1.2]
    labels=["No","No","Yes","Yes","Yes"]
    result=choose_validation_threshold(scores,labels)
    assert result["validation"]["f1"]==1.0
    assert result["validation_yes"]==3
    assert result["validation_no"]==2
    threshold=result["selected_threshold"]
    assert -1.0 < threshold <= -0.3
    untouched=summarize([-0.8,-0.2,0.9],["No","Yes","Yes"],threshold)
    assert untouched["count"]==3
    assert untouched["tp"]==2
    assert result["selection_policy"].endswith("no test labels used")


def test_fixed_uncalibrated_threshold_is_reproducible():
    report=summarize([-4,0.01,0.9,-0.1],["No","No","Yes","Yes"],0)
    assert (report["tp"],report["fp"],report["fn"],report["tn"])==(1,1,1,1)
    assert report["f1"]==pytest.approx(0.5)
    assert report["accuracy"]==pytest.approx(0.5)


def test_no_missing_class_or_nonfinite_scores():
    with pytest.raises(ValueError,match="both"):
        choose_validation_threshold([1,2],["No","No"])
    with pytest.raises(ValueError,match="finite"):
        check_qa_rows([math.nan,0.0],["Yes","No"])
    with pytest.raises(ValueError,match="finite"):
        check_qa_rows([math.inf,0.0],["Yes","No"])
    with pytest.raises(ValueError,match="Nonempty"):
        check_qa_rows([],[])
    with pytest.raises(ValueError,match="Yes/No"):
        check_qa_rows([0.0],["Maybe"])


def test_wrong_model_output_or_mismatched_metadata_rejected():
    ex=SimpleNamespace(answer="Yes",options=["Yes","No"],series=[],evaluated_question=lambda:"What?")
    class Stub:
        device=torch.device("cpu")
        def eval(self): pass
        def loss(self,series,prompt,answer,reduction):
            return torch.tensor([float("nan")])
    with pytest.raises(ValueError,match="Nonfinite"):
        score_option_margin(Stub(),ex)
    with pytest.raises(ValueError,match="metadata"):
        score_examples(Stub(),[ex],[],phase="validation")
    with pytest.raises(ValueError,match="disagrees"):
        score_examples(Stub(),[ex],[{"unit":1,"cycle":30,"answer":"No"}],phase="validation")


def test_scoring_keeps_true_engine_and_cycle_identifiers():
    ex=SimpleNamespace(answer="Yes",options=["Yes","No"],series=[],evaluated_question=lambda:"Q?")
    class Stub:
        device=torch.device("cpu")
        def eval(self):pass
        def loss(self,series,prompt,answer,reduction):
            return torch.tensor([0.5 if answer=="Yes" else 1.5])
    scores,records=score_examples(Stub(),[ex],[{"unit":3,"cycle":52,"answer":"Yes"}],phase="validation")
    assert scores==[1.0]
    assert records==[{"unit":3,"cycle":52,"actual":"Yes",
                      "yes_log_likelihood_margin":1.0,"phase":"validation"}]
