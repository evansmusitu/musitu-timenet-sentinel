"""Guard exact published model-perturbation evidence and page source."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"docs/evidence/sensor_sensitivity.json"
EXPECTED="d429498ea5def642bffeeee4caa3f9d2c6663039ca1e45bd012cc68449770338"


def test_public_sensor_sensitivity_equals_verified_cpu_artifact():
    payload=FILE.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==EXPECTED
    d=json.loads(payload)
    assert d["frozen_endpoint_predictions_reproduced"] is True
    assert d["recorded_case_predictions_reproduced"] is True
    assert d["not_opentime_5b"] is True
    assert d["heldout_engines_scored"]==100
    assert len(d["channels"])==24
    assert len({x["channel"] for x in d["channels"]})==24
    assert len(d["recorded_cases"])==6
    assert set(Counter(x["category"] for x in d["recorded_cases"]))=={
        "caught_risk","missed_risk","premature_alert","clean_monitoring"
    }
    for c in d["recorded_cases"]:
        p=c["model_warning_probability_before"]
        assert math.isfinite(p) and 0<=p<=1
        assert len(c["most_sensitive_channels"])==5
        abs_changes=[abs(x["risk_change_percentage_points"]) for x in c["most_sensitive_channels"]]
        assert abs_changes==sorted(abs_changes,reverse=True)
        for signal in c["most_sensitive_channels"]:
            q=signal["risk_after_channel_mean_replacement"]
            assert math.isfinite(q) and 0<=q<=1
            assert math.isclose(signal["risk_change_percentage_points"],100*(p-q),abs_tol=1e-6)
    importance=[x["mean_absolute_probability_change_percentage_points"] for x in d["channels"]]
    assert importance==sorted(importance,reverse=True)


def test_public_page_does_not_claim_causal_diagnosis_or_5b_results():
    html=(ROOT/"docs/sensor-sensitivity.html").read_text()
    for fragment in ("evidence/sensor_sensitivity.json","model sensitivity",
                     "NOT", "training-set mean", "not", "original"):
        assert fragment.lower() in html.lower()
    assert "sensor_13" not in html # The interface loads measured results, not hardcoded fabricated ranking
