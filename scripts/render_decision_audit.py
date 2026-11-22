"""Render the static held-out decision audit from the aggregate-only JSON.

No network access and no NASA per-engine rows are used or published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE_MARKER = "__AGGREGATE_AUDIT_DATA__"


def render(source: Path, template: Path, destination: Path) -> str:
    data = json.loads(source.read_text(encoding="utf-8"))
    assert data["test_population"]["independent_engine_endpoints"] == 100
    assert data["fixed_operating_point"]["test_labels_used_for_threshold_selection"] is False
    counts_a = data["baseline"]["confusion"]
    counts_b = data["compact_temporal_qa"]["confusion"]
    for counts in (counts_a, counts_b):
        assert sum(counts.values()) == 100
    html = template.read_text(encoding="utf-8")
    if html.count(TEMPLATE_MARKER) != 1:
        raise ValueError("Template must contain exactly one audit JSON marker")
    # Protect the enclosing script element. JSON is aggregate-only and authored by our audit script.
    literal = json.dumps(data, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace(TEMPLATE_MARKER, literal)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return html


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("docs/evidence/heldout_decision_audit.json"))
    p.add_argument("--template", type=Path, default=Path("docs/decision-audit.template.html"))
    p.add_argument("--output", type=Path, default=Path("docs/decision-audit.html"))
    args = p.parse_args()
    print(f"Rendered {len(render(args.source, args.template, args.output))} bytes to {args.output}")


if __name__ == "__main__":
    main()
