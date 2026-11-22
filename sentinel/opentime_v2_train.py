"""Full fine-tune of the OpenTime 5B checkpoint on all FD001 development engines (GPU only).

Protocol (decided before any test-set evaluation):
1. Train on every development-train engine (80) with full past-only histories; select the epoch
   with the lowest validation loss on 20 disjoint development-validation engines (upstream recipe).
2. Score each question as log P("Yes") - log P("No") under the model.
3. Choose the warning threshold that maximises F1 on truncated histories of the validation engines,
   which mimic NASA's test protocol. The official test set plays no part in steps 1-3.
4. Evaluate once on the 100 official NASA test engines: thresholded score, the model's untuned
   argmax (score >= 0) and its greedy generated answer.

No cloud provisioning or billing happens here.
"""
import argparse
import json
import random
import time
from pathlib import Path

from .data import download_dataset, load_fd001
from .opentime_examples import partitions
from .opentime_train import MODEL_ID, HELPER_REV, helper
from . import opentime_v2_examples as v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/opentime_v2"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-budget-seconds", type=float, default=2400,
                   help="Shrink the training set (balanced) if the measured speed cannot fit it")
    p.add_argument("--val-loss-limit", type=int, default=96)
    p.add_argument("--val-cuts-per-engine", type=int, default=10)
    args = p.parse_args()

    import re
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sklearn.metrics import roc_auc_score

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Supply an immutable checkpoint commit SHA")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    random.seed(42); torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train, test, labels = load_fd001(download_dataset(Path("data/raw")))
    dev_a, dev_b = partitions(train)
    v2.check_sensors(dev_a)
    train_rows = v2.training_rows(dev_a, seed=42)
    rng = random.Random(43)
    val_pool = v2.training_rows(dev_b, seed=43)
    val_loss_rows = rng.sample(val_pool, min(args.val_loss_limit, len(val_pool)))
    val_cut_rows = v2.truncated_rows(dev_b, args.val_cuts_per_engine, seed=44)
    test_rows = v2.official_rows(test, labels)

    u, helper_sha = helper(args.output / "reference_utils.py")
    path = snapshot_download(MODEL_ID, revision=args.revision,
                             allow_patterns=["*.json", "*.model", "*.safetensors"])
    model = u.OpenTSLMFlamingo.from_file(path, "cuda", torch.bfloat16)
    print(json.dumps({"event": "model_loaded", "seconds": round(time.time() - started)}), flush=True)

    def ex(r):
        series = [u.SeriesInput(values=s["values"], name=s["name"]) for s in r["series"]]
        return u.Example(series=series, prompt=r["prompt"], options=["Yes", "No"],
                         answer=r["answer"], category="warning_30")

    # Measure training speed on 6 real training examples, then fit the set to the time budget.
    probe = [ex(r) for r in train_rows[:6]]
    model.requires_grad_(False)
    for m in model.trained_modules():
        m.float().requires_grad_(True)
    torch.cuda.synchronize(); t0 = time.time()
    for e in probe:
        with u._autocast(model):
            loss = model.loss(e.series, e.question(random.Random(0)), e.answer)
        loss.backward()
    torch.cuda.synchronize()
    per_example = (time.time() - t0) / len(probe)
    model.zero_grad(set_to_none=True)
    fit = int(args.train_budget_seconds / (per_example * args.epochs * 1.15))
    if fit < len(train_rows):
        yes = [r for r in train_rows if r["answer"] == "Yes"]
        no = [r for r in train_rows if r["answer"] == "No"]
        k = max(8, fit // 2)
        train_rows = random.Random(45).sample(yes, min(k, len(yes))) + random.Random(46).sample(no, min(k, len(no)))
        random.Random(47).shuffle(train_rows)
    print(json.dumps({"event": "train_plan", "seconds_per_example": round(per_example, 3),
                      "train_examples": len(train_rows), "epochs": args.epochs}), flush=True)

    tr = [ex(r) for r in train_rows]
    vl = [ex(r) for r in val_loss_rows]
    t_train = time.time()
    u.train(model, tr, vl, epochs=args.epochs, accumulation=8, seed=42)
    train_seconds = time.time() - t_train
    model.eval()

    @torch.inference_mode()
    def score(r):
        e = ex(r)
        q = e.evaluated_question()
        with u._autocast(model):
            yes = float(model.loss(e.series, q, "Yes", reduction="none").sum())
            no = float(model.loss(e.series, q, "No", reduction="none").sum())
        return no - yes  # log P(Yes) - log P(No), summed over answer and end tokens

    val_scores = [score(r) for r in val_cut_rows]
    val_actual = [r["answer"] == "Yes" for r in val_cut_rows]
    threshold, val_f1 = v2.best_threshold(val_scores, val_actual)

    out = []
    for r in test_rows:
        s = score(r)
        e = ex(r)
        with u._autocast(model):
            raw = model.generate(e.series, e.evaluated_question(), max_new_tokens=8)
        out.append({"unit": r["unit"], "cycle": r["cycle"], "rul": r["rul"], "actual": r["answer"],
                    "score": round(s, 4), "generated": u.chosen_option(raw, e.options), "raw": raw[:40]})
    actual = np.array([x["actual"] == "Yes" for x in out])
    scores = np.array([x["score"] for x in out])
    gen = [x["generated"] for x in out]
    gen_pred = [(g == "Yes") if g is not None else (not a) for g, a in zip(gen, actual)]  # invalid = wrong

    report = {
        "checkpoint": MODEL_ID, "revision": args.revision, "helper_revision": HELPER_REV,
        "helper_sha256": helper_sha, "dataset": "NASA C-MAPSS FD001 (simulated)",
        "protocol": "full past-only history, 14 varying sensors, cycle count in prompt; epoch by "
                    "validation loss; threshold by F1 on validation-engine truncations; one test pass",
        "train_examples": len(tr), "val_loss_examples": len(vl), "val_threshold_examples": len(val_cut_rows),
        "train_engines": int(dev_a.unit.nunique()), "val_engines": int(dev_b.unit.nunique()),
        "epochs": args.epochs, "seconds_per_example_probe": per_example, "train_seconds": train_seconds,
        "threshold_from_validation": threshold, "validation_f1_at_threshold": val_f1,
        "validation_auroc": float(roc_auc_score(val_actual, val_scores)) if 0 < sum(val_actual) < len(val_actual) else None,
        "test": {
            "count": len(out), "positives": int(actual.sum()),
            "auroc": float(roc_auc_score(actual, scores)),
            "validation_threshold": v2.confusion(scores >= threshold, actual),
            "untuned_argmax": v2.confusion(scores >= 0, actual),
            "greedy_generation": {**v2.confusion(gen_pred, actual),
                                  "invalid_outputs": sum(g is None for g in gen)},
        },
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "total_seconds": time.time() - started,
        "limits": "Simulated engines; binary 30-cycle warning; invalid generations scored as errors.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "test_predictions.json").write_text(json.dumps(out, indent=1))
    print("MUSITU_V2_METRICS " + json.dumps(report), flush=True)
    for x in out:  # one line each, so log collection cannot truncate a long line
        print("MUSITU_V2_PRED " + json.dumps(x), flush=True)
    print("MUSITU_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
