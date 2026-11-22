"""Prevalence-preserving, engine-diverse validation sampling for OpenTime v2.

Labels are taken only from development engines; official test labels never enter
this sampler. The training sampler may still balance its training examples.
"""
from __future__ import annotations

import random
from collections import defaultdict


def prevalence_matched_engine_diverse(rows: list[dict], limit: int, *, seed: int = 43) -> list[dict]:
    """Sample validation windows with natural class ratio and engine coverage.

    Allocate positive examples in proportion to their source prevalence (rounded
    to the nearest integer), then sample each class across distinct engines before
    drawing a second window from the same engine. The result is deterministic.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 2:
        raise ValueError("validation limit must be an integer >= 2")
    if not rows or limit > len(rows):
        raise ValueError("validation limit exceeds available source windows")
    seen: set[tuple[int, int]] = set()
    pools: dict[str, dict[int, list[dict]]] = {
        "Yes": defaultdict(list), "No": defaultdict(list)
    }
    for row in rows:
        unit, cycle, answer = row.get("unit"), row.get("cycle"), row.get("answer")
        if (type(unit) is not int or unit <= 0 or
                type(cycle) is not int or cycle <= 0):
            raise ValueError("engine and cycle IDs must be positive integers")
        if answer not in pools:
            raise ValueError("validation answer must be Yes or No")
        if (unit, cycle) in seen:
            raise ValueError("duplicate engine/cycle validation window")
        seen.add((unit, cycle))
        pools[answer][unit].append(row)
    yes_available = sum(map(len, pools["Yes"].values()))
    no_available = sum(map(len, pools["No"].values()))
    if min(yes_available, no_available) == 0:
        raise ValueError("validation needs both Yes and No classes")
    yes_count = max(1, min(limit - 1, (limit * yes_available + len(rows) // 2) // len(rows)))
    # Respect class availability, without changing the total requested count.
    yes_count = max(limit - no_available, min(yes_count, yes_available))
    if not 0 < yes_count < limit:
        raise ValueError("cannot draw both classes at requested validation limit")
    target = {"Yes": yes_count, "No": limit - yes_count}
    rng = random.Random(seed)
    selected: list[dict] = []
    for label in ("Yes", "No"):
        per_engine = []
        for engine, engine_rows in sorted(pools[label].items()):
            group = list(engine_rows)
            rng.shuffle(group)
            per_engine.append(group)
        rng.shuffle(per_engine)
        added = 0
        while added < target[label]:
            progressed = False
            for group in per_engine:
                if added >= target[label]:
                    break
                if group:
                    selected.append(group.pop())
                    added += 1
                    progressed = True
            if not progressed:
                raise ValueError("validation sample exhausted unexpectedly")
    rng.shuffle(selected)
    if len(selected) != limit:
        raise AssertionError("sampling size invariant violated")
    return selected
