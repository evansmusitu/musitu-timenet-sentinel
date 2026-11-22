"""Offline checks for the static public, aggregate-only held-out dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.render_decision_audit import render


def audit_fixture():
    return {
        "test_population": {"independent_engine_endpoints": 100},
        "fixed_operating_point": {"test_labels_used_for_threshold_selection": False},
        "baseline": {"confusion": {"tp": 20, "fp": 0, "fn": 5, "tn": 75}},
        "compact_temporal_qa": {"confusion": {"tp": 24, "fp": 1, "fn": 1, "tn": 74}},
    }


def test_render_inlines_aggregate_json_without_network(tmp_path):
    source, template, dest = (tmp_path / n for n in ("report.json", "template.html", "decision-audit.html"))
    source.write_text(json.dumps(audit_fixture()))
    template.write_text("<script type='application/json'>__AGGREGATE_AUDIT_DATA__</script>")
    rendered = render(source, template, dest)
    assert dest.read_text() == rendered
    assert '"tp":20' in rendered
    assert "__AGGREGATE_AUDIT_DATA__" not in rendered
    assert rendered == render(source, template, dest)


def test_render_rejects_missing_template_marker(tmp_path):
    source, template, dest = (tmp_path / n for n in ("report.json", "template.html", "decision-audit.html"))
    source.write_text(json.dumps(audit_fixture()))
    template.write_text("<html>no placeholder</html>")
    with pytest.raises(ValueError, match="exactly one"):
        render(source, template, dest)


def test_public_dashboard_has_sources_and_benchmarks():
    root = Path(__file__).resolve().parents[1]
    template = (root / "docs/decision-audit.template.html").read_text()
    report = json.loads((root / "docs/evidence/heldout_decision_audit.json").read_text())
    assert template.count("__AGGREGATE_AUDIT_DATA__") == 1
    assert "missratio" in template and "input type=\"range\"" in template
    assert "not an aircraft safety claim" in template
    assert "Not a 5B victory claim" in template
    assert "449" not in str(report["test_population"])
    assert report["baseline"]["confusion"] == {"tp": 20, "fp": 0, "fn": 5, "tn": 75}
    assert report["compact_temporal_qa"]["confusion"] == {"tp": 24, "fp": 1, "fn": 1, "tn": 74}
    assert report["bootstrap_uncertainty"]["temporal_minus_baseline_warning_f1"]["lower"] < 0
    assert report["bootstrap_uncertainty"]["temporal_minus_baseline_warning_f1"]["upper"] > 0
    assert report["provenance"]["baseline_predictions_sha256"] and report["provenance"]["compact_predictions_sha256"]
