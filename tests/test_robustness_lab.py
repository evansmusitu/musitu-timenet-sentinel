"""Freeze provenance and reject misleading domain-shift comparison claims."""
import json
from pathlib import Path
from scripts.render_robustness_lab import build_aggregate,render

BASE=Path(__file__).resolve().parents[1]
REPORTS=BASE/'docs/evidence'


def _load():
    return [json.loads((REPORTS/name).read_text()) for name in
        ['frozen_classical_cross_cohort.json','frozen_compact_cross_cohort.json','multidomain_fd004_exploratory.json','multidomain_fd004_classifier.json']]


def test_aggregates_have_one_consistent_official_test_population():
    data=build_aggregate(*_load())
    assert [row['cohort'] for row in data['rows']]==['FD001','FD002','FD003','FD004']
    assert [r['n'] for r in data['rows']]==[100,253,100,237]
    assert data['rows'][3]['compact_confusion']['fp']==112
    assert data['multi_domain_fd004']['confusion']['fn']==26
    assert data['multi_domain_warning_classifier_fd004']['tp']==40
    assert 'simulated' in data['scientific_scope'].lower()


def test_static_report_renders_deterministically_and_mentions_limits():
    data=build_aggregate(*_load())
    tmpl=BASE/'docs/robustness-lab.template.html'
    result=render(data,tmpl)
    assert result==render(data,tmpl)
    assert '<table' in result and '0.479' in result and '0.643' in result and '0.816' in result
    assert 'post-hoc' in result and 'do not</strong> establish cross-manufacturer generalization' in result
    assert '$bars' not in result
    assert '<script' not in result


def test_public_source_uses_same_frozen_model_ids():
    data=build_aggregate(*_load())
    assert '45003875499' in data['provenance']['original_classical']
    assert '45003991598' in data['provenance']['original_compact']
    assert '45004350622' in data['provenance']['exploratory_multidomain']
    assert '45004601674' in data['provenance']['exploratory_classifier']
