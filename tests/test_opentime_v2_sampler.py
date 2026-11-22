"""CPU-only tests for prevalence-matched, engine-disjoint GPU validation sampling."""
from collections import Counter
import pytest
from sentinel.opentime_v2_sampler import prevalence_matched_engine_diverse as sample


def population():
    # 20 independent development engines, 1 positive / 5 negative each.
    return [dict(unit=u, cycle=c, answer="Yes" if c == 6 else "No")
            for u in range(1, 21) for c in range(1, 7)]


def test_prevalence_matched_engine_diverse_and_reproducible():
    rows = population()
    chosen = sample(rows, 24, seed=43)
    assert len(chosen) == 24
    assert Counter(r["answer"] for r in chosen) == {"Yes": 4, "No": 20}
    assert len({r["unit"] for r in chosen}) == 20
    assert chosen == sample(rows, 24, seed=43)
    assert len({(r["unit"], r["cycle"]) for r in chosen}) == 24


def test_not_balanced_at_natural_prevalence():
    chosen = sample(population(), 60, seed=43)
    assert Counter(r["answer"] for r in chosen) == {"Yes": 10, "No": 50}


@pytest.mark.parametrize("rows,limit,match", [
    ([], 2, "exceeds"),
    ([{"unit": 1, "cycle": 1, "answer": "Yes"}], 2, "exceeds"),
    ([{"unit": 1, "cycle": 1, "answer": "Yes"}] * 2, 2, "duplicate"),
    ([{"unit": True, "cycle": 1, "answer": "Yes"},
      {"unit": 2, "cycle": 1, "answer": "No"}], 2, "positive integers"),
    ([{"unit": 1, "cycle": 1, "answer": "Maybe"},
      {"unit": 2, "cycle": 1, "answer": "No"}], 2, "Yes or No"),
    ([{"unit": 1, "cycle": 1, "answer": "Yes"},
      {"unit": 2, "cycle": 1, "answer": "Yes"}], 2, "both Yes and No"),
])
def test_malformed_validation_fails_closed(rows, limit, match):
    with pytest.raises(ValueError, match=match):
        sample(rows, limit)


def test_rejects_bool_limit():
    with pytest.raises(ValueError, match="integer"):
        sample(population(), True)
