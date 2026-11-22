"""Contract and audit of published paired longitudinal evaluation evidence."""
import json
from pathlib import Path
import pytest


def test_published_paired_evidence_is_consistent_with_original_2163_window_protocol():
    p=Path(__file__).resolve().parents[1]/"docs/evidence/paired_replay_comparison.json"
    report=json.loads(p.read_text())
    assert report["same_windows_and_labels"] is True
    assert report["baseline_original_100_endpoints_reproduced"] is True
    assert report["compact_original_100_endpoints_reproduced_by_source_run"] is True
    assert "NOT OpenTime 5B" in report["compact_model"]
    assert report["original_baseline_predictions_sha256"]=="eb6ba0cbc4007ea32c8da1ce27b8e4818ae3a53f217129975aa9c638bf5cccd3"
    assert report["original_compact_replay_predictions_sha256"]=="5c861bb149562571d3fefe72541ab38355596816184816c8a8b72c88d944114d"
    for model in ("compact","classical"):
        z=report[model]
        a=z["window_confusion"]
        assert sum(a.values())==2163
        assert z["window_warning_f1"]==pytest.approx(2*a["tp"]/(2*a["tp"]+a["fp"]+a["fn"]))
        assert z["engines"]["total"]==100
        assert z["engines"]["eligible"]+z["engines"]["never_observed_within_horizon"]==100
        assert z["engines"]["detected"]<=z["engines"]["eligible"]
        assert z["engines"]["premature_alert"]<=100
    x=report["paired_engine_differences"]
    assert (x["both_detect_risk"]+x["compact_only_risk_detection"]
            +x["baseline_only_risk_detection"]+x["neither_detect_risk"])==25
    assert (x["both_detect_risk"]+x["compact_only_risk_detection"])==report["compact"]["engines"]["detected"]
    assert (x["both_detect_risk"]+x["baseline_only_risk_detection"])==report["classical"]["engines"]["detected"]
    site=(p.parents[1]/"maintenance-replay.html").read_text()
    assert "24 vs 21" in site and "8 vs 5" in site
    assert "not statistically conclusive" in site
