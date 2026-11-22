"""Option-likelihood evaluation of a genuine OpenTime binary QA model.

Validation-only threshold selection, frozen model at test evaluation, and a
separate report from generated Yes/No answers. This does not train a model or
claim improved performance on any held-out engines.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def check_qa_rows(margins: Sequence[float], answers: Sequence[str]) -> None:
    if len(margins) != len(answers) or not answers:
        raise ValueError("Nonempty, equally sized scores and Yes/No answers required")
    if any(type(y) is not str or y not in ("Yes", "No") for y in answers):
        raise ValueError("Only Yes/No ground truth allowed")
    if any(not isinstance(s, (int, float)) or isinstance(s, bool)
           or not math.isfinite(float(s)) for s in margins):
        raise ValueError("Log-likelihood margins must be finite numbers")


def summarize(margins: Sequence[float], answers: Sequence[str], threshold: float) -> dict:
    """Decision confusion matrix using the same fixed threshold for every row."""
    check_qa_rows(margins, answers)
    if not math.isfinite(threshold):
        raise ValueError("Nonfinite decision threshold")
    tp = fp = fn = tn = 0
    for score, answer in zip(margins, answers, strict=True):
        yes = score >= threshold
        if answer == "Yes":
            if yes:
                tp += 1
            else:
                fn += 1
        else:
            if yes:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "count": len(answers),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(answers),
        "precision": precision, "recall": recall, "f1": f1,
    }


def choose_validation_threshold(margins: Sequence[float], answers: Sequence[str]) -> dict:
    """Pick a threshold using development validation labels ONLY.

    Maximize validation F1, then prefer the option-equivalent threshold 0.0
    on ties. The identical selected threshold is later used on untouched test
    engines. This is exploratory and can overfit a small validation cohort.
    """
    check_qa_rows(margins, answers)
    counts = Counter(answers)
    if not counts["Yes"] or not counts["No"]:
        raise ValueError("Validation must contain both warning classes")
    ordered = sorted(set(float(s) for s in margins))
    boundaries = [math.nextafter(ordered[0], -math.inf)]
    boundaries += [(a + b) / 2 for a, b in zip(ordered[:-1], ordered[1:])]
    boundaries += [math.nextafter(ordered[-1], math.inf), 0.0]
    candidates = [x for x in boundaries if math.isfinite(x)]
    if not candidates:
        raise ValueError("No valid threshold candidates")
    best = min(
        candidates,
        key=lambda threshold: (
            -summarize(margins, answers, threshold)["f1"],
            abs(threshold),
            threshold,
        ),
    )
    return {
        "selected_threshold": float(best),
        "validation": summarize(margins, answers, best),
        "validation_yes": counts["Yes"],
        "validation_no": counts["No"],
        "selection_policy": "max_validation_F1_then_min_abs_threshold; no test labels used",
        "uncalibrated_validation": summarize(margins, answers, 0.0),
    }


def score_option_margin(model, example) -> float:
    """Compare negative log-likelihood of the exact Yes and No answer strings.

    Score = NLL(No) - NLL(Yes), with the SAME option-presented prompt for
    both candidates. Positive favors Yes. Includes model-defined EOS cost.
    """
    import torch

    model.eval()
    prompt = example.evaluated_question()
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Missing QA prompt")
    if set(example.options) != {"Yes", "No"}:
        raise ValueError("Only exactly Yes/No options supported")
    values = {}
    with torch.inference_mode():
        for answer in ("Yes", "No"):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=model.device.type == "cuda"):
                loss = model.loss(example.series, prompt, answer, reduction="none")
            if loss.ndim != 1 or loss.numel() < 1 or not torch.isfinite(loss).all():
                raise ValueError("Nonfinite or malformed candidate-token losses")
            values[answer] = float(loss.float().sum())
    margin = values["No"] - values["Yes"]
    if not math.isfinite(margin):
        raise ValueError("Nonfinite option likelihood margin")
    return margin


def score_examples(model, examples: Sequence, rows: Sequence[dict], *, phase: str) -> tuple[list[float], list[dict]]:
    if len(examples) != len(rows) or not examples:
        raise ValueError("Model examples and metadata must match")
    scores, results = [], []
    for ex, row in zip(examples, rows, strict=True):
        if row.get("answer") not in ("Yes", "No"):
            raise ValueError("Invalid row answer")
        if ex.answer != row["answer"]:
            raise ValueError("Model example truth disagrees with original row")
        if type(row.get("unit")) is not int or type(row.get("cycle")) is not int:
            raise ValueError("Missing original engine and cycle identities")
        score = score_option_margin(model, ex)
        scores.append(score)
        results.append({
            "unit": row["unit"], "cycle": row["cycle"],
            "actual": row["answer"], "yes_log_likelihood_margin": score,
            "phase": phase,
        })
    if len({(r["unit"], r["cycle"]) for r in results}) != len(results):
        raise ValueError("Repeated engine/cycle in model evaluation")
    return scores, results
