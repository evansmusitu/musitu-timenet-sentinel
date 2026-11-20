"""Train a small prompt-limited temporal question-answering Transformer on NASA FD001.

This is an original, compact, fixed-task experimental model. It is NOT
OpenTime's 5B checkpoint, NOT a general-purpose language model, and its outputs
are for the NASA simulated benchmark only.

Two predefined questions:
  "How many operating cycles remain?" -> a discrete 10-cycle answer token
  "Could this engine fail within 30 cycles?" -> YES / NO answer token

Sensor samples are past-only. Engine IDs are group-isolated and never features.
Official test labels are read only at the final evaluation stage.
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .data import COLS, download_dataset, load_fd001
from .features import split_engine_ids
from .benchmark import metrics

RUL_CENTERS = np.asarray([5.0 + 10.0 * i for i in range(21)], dtype=np.float32)
TASKS = {
    "remaining_life": "How many operating cycles remain?",
    "warning_30": "Could this engine fail within 30 cycles?",
}

def windows(frame, engine_ids, *, official_rul=None, window=30, stride=10):
    """Generate windows and labels from disjoint engine groups."""
    xs, ys, meta = [], [], []
    for unit, group in frame.groupby("unit", sort=True):
        if int(unit) not in engine_ids:
            continue
        group = group.sort_values("cycle").reset_index(drop=True)
        if len(group) < window:
            continue
        if official_rul is None:
            ends = list(range(window - 1, len(group), stride))
            failure_cycle = int(group.cycle.iloc[-1])
        else:
            if int(unit) not in official_rul:
                raise ValueError("Missing official test label")
            ends = [len(group) - 1]
        for end in ends:
            age = int(group.cycle.iloc[end])
            history = group.loc[end-window+1:end, COLS[2:]].to_numpy(dtype=np.float32)
            target = failure_cycle - age if official_rul is None else int(official_rul[int(unit)])
            xs.append(history)
            ys.append(float(target))
            meta.append((int(unit), age))
    if not xs:
        raise ValueError("No valid windows for dataset")
    return np.stack(xs), np.array(ys, dtype=np.float32), meta

def build_model():
    import torch
    from torch import nn

    class TemporalQATransformer(nn.Module):
        def __init__(self):
            super().__init__()
            width = 48
            self.proj = nn.Linear(24, width)
            self.position = nn.Parameter(torch.zeros(1, 31, width))
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=96, dropout=0.1,
                batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2,
                                                 enable_nested_tensor=False)
            self.task_embedding = nn.Embedding(2, width)
            self.rul_head = nn.Linear(width, 21)
            self.warning_head = nn.Linear(width, 2)

        def forward(self, x):
            h = self.proj(x)
            cls = self.cls.expand(x.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1) + self.position
            pooled = self.encoder(h)[:, 0, :]
            q_rul = pooled + self.task_embedding(
                torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            q_warn = pooled + self.task_embedding(
                torch.ones(x.shape[0], dtype=torch.long, device=x.device))
            return self.rul_head(q_rul), self.warning_head(q_warn)

    return TemporalQATransformer()

def predict(model, x, norm, batch_size=256):
    import torch
    mean, scale = norm
    x = (x - mean) / scale
    model.eval()
    predictions, warnings, probabilities = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits_rul, logits_warn = model(torch.from_numpy(x[start:start+batch_size].astype("float32")))
            p = logits_rul.softmax(dim=1).cpu().numpy()
            predictions.extend((p * RUL_CENTERS[None, :]).sum(axis=1).tolist())
            risk = logits_warn.softmax(dim=1)[:,1].cpu().numpy()
            probabilities.extend(risk.tolist())
            warnings.extend((risk >= 0.5).tolist())
    return np.array(predictions), np.array(warnings), np.array(probabilities)

def run(data_dir: Path, output_dir: Path, *, download=False, epochs=12):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if download:
        download_dataset(data_dir)
    train_df, test_df, labels = load_fd001(data_dir)
    groups = pd.DataFrame({"unit": sorted(train_df.unit.unique()),
                           "split_source": "development"})
    train_ids, val_ids = split_engine_ids(groups)
    xtrain, ytrain, train_meta = windows(train_df, set(train_ids))
    xval, yval, val_meta = windows(train_df, set(val_ids))
    xtest, ytest, test_meta = windows(test_df, set(test_df.unit.unique()), official_rul=labels)
    mean = xtrain.mean(axis=(0,1), keepdims=True)
    std = xtrain.std(axis=(0,1), keepdims=True)
    scale = np.maximum(std, 1e-5)
    x_norm = ((xtrain - mean) / scale).astype("float32")
    ybins = np.clip(np.floor(ytrain / 10), 0, 20).astype("int64")
    ywarn = (ytrain <= 30).astype("int64")
    dataset = TensorDataset(torch.from_numpy(x_norm),
                            torch.from_numpy(ybins),
                            torch.from_numpy(ywarn))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, bins, warning in loader:
            out_bin, out_warn = model(xb)
            loss = F.cross_entropy(out_bin, bins) + 0.4*F.cross_entropy(out_warn, warning)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())*len(xb)
        valpred, _, _ = predict(model, xval, (mean,scale))
        val_mae = float(np.mean(np.abs(valpred-yval)))
        history.append({"epoch": epoch+1, "train_loss": total_loss/len(dataset), "validation_rul_mae": val_mae})
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f} validation_mae={val_mae:.3f}",flush=True)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    valpred, valwarning, valrisk = predict(model, xval, (mean,scale))
    # Reserved official held-out test engines are scored only after training/selection.
    testpred, testwarning, testrisk = predict(model, xtest, (mean,scale))

    def binary_scores(y_true, y_warn):
        y = y_true <= 30
        true_pos = int(np.sum(y_warn & y))
        false_pos = int(np.sum(y_warn & ~y))
        false_neg = int(np.sum(~y_warn & y))
        p = true_pos/(true_pos+false_pos) if true_pos+false_pos else 0.0
        r = true_pos/(true_pos+false_neg) if true_pos+false_neg else 0.0
        return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0}

    def scores(y, yhat, warn):
        data = metrics(y, yhat)
        data["trained_warning_head"] = binary_scores(y,warn)
        return data

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "project":"MUSITU AXIOM Sentinel",
        "model":"Small prompt-limited temporal question-answering Transformer (from scratch)",
        "is_opentime_5b":False, "general_purpose_tslm":False,
        "questions":TASKS,
        "dataset":"NASA C-MAPSS FD001 simulated turbofan degradation",
        "training_engines":len(train_ids), "validation_engines":len(val_ids),
        "heldout_test_engines":len(test_meta), "train_windows":len(xtrain),
        "validation_windows":len(xval), "training_epochs":epochs,
        "best_validation_mae":best_val, "history":history,
        "validation":scores(yval,valpred,valwarning),
        "official_test":scores(ytest,testpred,testwarning),
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "limits":"Simulated benchmark only; small fixed-prompt model, not the experimental OpenTime 5B checkpoint."
    }
    (output_dir / "temporal_qa_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    df = pd.DataFrame(test_meta,columns=["unit","observed_cycle"])
    df["actual_rul"]=ytest
    df["predicted_rul"]=testpred
    df["risk_within_30_probability"]=testrisk
    df["risk_within_30_predicted"]=testwarning
    df.to_csv(output_dir/"temporal_qa_predictions.csv",index=False)
    torch.save({"model_state":best_state,"mean":mean,"scale":scale,
                "questions":TASKS,"answer_tokens":"RUL_BIN_000_to_200, YES, NO"},
               output_dir/"temporal_qa_checkpoint.pt")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=Path("data/raw"))
    p.add_argument("--output-dir",type=Path,default=Path("artifacts"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--epochs",type=int,default=12)
    a=p.parse_args()
    if not 1<=a.epochs<=50:
        p.error("epochs must be between 1 and 50")
    run(a.data_dir,a.output_dir,download=a.download,epochs=a.epochs)

if __name__=="__main__":
    main()
