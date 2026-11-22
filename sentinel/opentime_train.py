"""GPU-only OpenTime checkpoint adaptation. No cloud provisioning or billing."""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from .data import download_dataset, load_fd001
from .opentime_examples import examples, partitions, QUESTION

MODEL_ID="opentime/tslm-nov26"
HELPER_REV="b83a9901d401e9e872ee9048fe61e6c9df9a5b78"
HELPER_SHA256="c19db31d2ce6e84a4ce02c9a0fdfe0869c53775f77b04dd4be5c893afd8335e2"
HELPER_URL=f"https://raw.githubusercontent.com/aws-samples/generative-ai-on-amazon-sagemaker/{HELPER_REV}/examples/opentslm-finetune-arfbench/scripts/utils.py"

def helper(path):
    # Execute only the pinned organizer-recommended open-source implementation.
    with urlopen(HELPER_URL,timeout=60) as response:
        data=response.read(2_000_001)
    if len(data)>2_000_000:
        raise ValueError("Unexpected helper size")
    actual_hash=hashlib.sha256(data).hexdigest()
    if actual_hash!=HELPER_SHA256:
        raise ValueError("Pinned OpenTSLM training helper SHA256 mismatch")
    path.write_bytes(data)
    spec=importlib.util.spec_from_file_location("opentime_reference",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module,hashlib.sha256(data).hexdigest()

def balanced(rows,limit,seed):
    # V2 training-only protocol: spread each class across distinct engines first.
    # Past GPU pilot used uniform row sampling and its published metrics are unchanged.
    from .opentime_sampling import balanced_engine_diverse
    return balanced_engine_diverse(rows,limit,seed=seed)

def validation_sample(rows, limit, seed, mode):
    if mode == "balanced":
        return balanced(rows, limit, seed)
    if mode == "prevalence_matched":
        from .opentime_v2_sampler import prevalence_matched_engine_diverse
        return prevalence_matched_engine_diverse(rows, limit, seed=seed)
    raise ValueError("Unsupported validation sampling mode")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",required=True,help="Immutable Hugging Face checkpoint commit SHA")
    p.add_argument("--output",type=Path,default=Path("artifacts/opentime"))
    p.add_argument("--train-limit",type=int,default=48)
    p.add_argument("--val-limit",type=int,default=24)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--option-likelihood", action="store_true",
                   help="After training, calibrate exact Yes/No answer token likelihoods on validation engines only, then score untouched official test engines")
    p.add_argument("--validation-sampling", choices=["balanced", "prevalence_matched"],
                   default="balanced", help="V2 uses prevalence_matched; never select by official test performance")
    args=p.parse_args()
    import re
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
    if not re.fullmatch(r"[0-9a-f]{40}",args.revision):
        raise ValueError("Supply an immutable checkpoint commit SHA")
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<45*1024**3:
        raise SystemExit("CUDA GPU with at least 48 GB advertised VRAM required; recommend 80 GB.")
    if args.epochs<1 or args.train_limit<2 or args.val_limit<2:
        raise ValueError("Invalid training parameters")
    random.seed(42); torch.manual_seed(42)
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    train,test,labels=load_fd001(download_dataset(Path("data/raw")))
    a,b=partitions(train)
    t=balanced(examples(a),args.train_limit,42)
    v=validation_sample(examples(b),args.val_limit,43,args.validation_sampling)
    from .opentime_sampling import audit_engine_groups
    cohort=audit_engine_groups(t,v)
    if len(t)!=args.train_limit or len(v)!=args.val_limit:
        raise ValueError("Training/validation size shortfall: refuse to launch adaptation")

    held=examples(test,official_rul=labels)
    assert len(held)==len(labels)
    u,sha=helper(args.output/"reference_utils.py")
    path=snapshot_download(MODEL_ID,revision=args.revision,allow_patterns=["*.json","*.model","*.safetensors"])
    model=u.OpenTSLMFlamingo.from_file(path,"cuda",torch.bfloat16)
    def as_example(row):
        series=[u.SeriesInput(**s) for s in row["series"]]
        if len(series)>model.max_media or any(len(s.values)>model.max_values for s in series):
            raise ValueError("Checkpoint does not support telemetry dimensions")
        return u.Example(series=series,prompt=QUESTION,options=["Yes","No"],answer=row["answer"],category="warning_30")
    tr,va,te=map(lambda rows:[as_example(x) for x in rows],(t,v,held))
    def evaluate(items,rows,name):
        model.eval()
        answers=[]
        for ex,row in zip(items,rows):
            raw=model.generate(ex.series,ex.evaluated_question(),max_new_tokens=12)
            chosen=u.chosen_option(raw,ex.options)
            # Invalid outputs are counted as incorrect, not silently mapped to No.
            answers.append({"unit":row["unit"],"cycle":row["cycle"],"actual":ex.answer,"prediction":chosen,"raw":raw})
        actual=[x["actual"]=="Yes" for x in answers]
        predicted=[(x["prediction"]=="Yes") if x["prediction"] is not None else not y for x,y in zip(answers,actual)]
        p,r,f,_=precision_recall_fscore_support(actual,predicted,average="binary",zero_division=0)
        (args.output/f"{name}_predictions.json").write_text(json.dumps(answers,indent=2))
        return {"count":len(answers),"accuracy":sum(x["actual"]==x["prediction"] for x in answers)/len(answers),
                "balanced_accuracy_invalid_as_error":float(balanced_accuracy_score(actual,predicted)),
                "precision_invalid_as_error":float(p),"recall_invalid_as_error":float(r),"f1_invalid_as_error":float(f),
                "invalid_outputs":sum(x["prediction"] is None for x in answers)}
    # Run both original and fine-tuned model on the locked test protocol. No test-driven selection.
    before=evaluate(te,held,"before")
    u.train(model,tr,va,epochs=args.epochs,accumulation=4,seed=42)
    after=evaluate(te,held,"after")
    likelihood_evaluation=None
    if args.option_likelihood:
        from .opentime_decision import choose_validation_threshold, score_examples, summarize
        # Threshold is selected ONLY on disjoint development validation engines.
        val_margins, val_rows=score_examples(model,va,v,phase="development_validation")
        tuned=choose_validation_threshold(val_margins,[r["answer"] for r in v])
        threshold=tuned["selected_threshold"]
        # The official test labels are read only AFTER the threshold is frozen.
        test_margins, test_rows=score_examples(model,te,held,phase="official_test")
        for row in test_rows:
            row["predicted_at_fixed_zero"]=("Yes" if row["yes_log_likelihood_margin"]>=0 else "No")
            row["predicted_at_validation_threshold"]=("Yes" if row["yes_log_likelihood_margin"]>=threshold else "No")
        (args.output/"validation_likelihood_predictions.json").write_text(json.dumps(val_rows,indent=2)+"\n")
        (args.output/"official_test_likelihood_predictions.json").write_text(json.dumps(test_rows,indent=2)+"\n")
        likelihood_evaluation={
            "approach":"genuine OpenTime 5B conditional answer-token negative log-likelihood",
            "margin_definition":"NLL(No) minus NLL(Yes); positive favors Yes",
            "validation_engine_holdout":True,
            "official_test_used_for_threshold_selection":False,
            "validation_threshold_selection":tuned,
            "test_uncalibrated_zero_threshold":summarize(test_margins,[r["answer"] for r in held],0.0),
            "test_validation_selected_threshold":summarize(test_margins,[r["answer"] for r in held],threshold),
        }
    weights={name:param.detach().cpu().contiguous() for name,param in model.named_parameters() if param.requires_grad}
    save_file(weights,args.output/"trained_parameters.safetensors")
    report={"checkpoint":MODEL_ID,"revision":args.revision,"helper_revision":HELPER_REV,"helper_sha256":sha,
        "dataset":"NASA C-MAPSS FD001","train_examples":len(tr),"validation_examples":len(va),
        "train_engine_count":int(a.unit.nunique()),"validation_engine_count":int(b.unit.nunique()),
        "training_sampling_protocol":"balanced-engine-diverse-v1 (after the original 16-example pilot)",
        "validation_sampling_protocol":args.validation_sampling,
        "selected_train_class_counts":cohort["train_class_counts"],
        "selected_validation_class_counts":cohort["validation_class_counts"],
        "selected_train_unique_engines":len({row["unit"] for row in t}),
        "selected_validation_unique_engines":len({row["unit"] for row in v}),
        "before":before,"after":after,
        "likelihood_evaluation":likelihood_evaluation,
        "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
        "epochs":args.epochs,"seconds":time.time()-started,
        "limits":"Simulation only; binary warning QA; invalid outputs scored as errors."}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()
