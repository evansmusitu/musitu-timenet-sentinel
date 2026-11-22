# Verified OpenTime November 2026 5B GPU results — NASA FD001

**Current result (v2, 22 November 2026):** the organizer's `opentime/tslm-nov26` checkpoint, fine-tuned on all 80 FD001 development-train engines, beats our classical gradient-boosting baseline on the 100 official NASA test engines in all three training seeds, and is on par with our separate compact temporal Transformer. The earlier 16-example pilot (F1 0.324) is kept below as **superseded history**.

## v2 results on the 100 official test engines (25 at risk within 30 cycles)

Primary metric, declared before testing: warning F1 at the threshold chosen on validation engines.

| System | Seed | Training examples | Caught | False alarms | Missed | Warning F1 (primary) | F1 of the model's own Yes/No answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical gradient boosting (baseline) | – | – | 20 | 0 | 5 | 0.889 | – |
| Compact temporal QA Transformer (not OpenTime) | – | – | 24 | 1 | 1 | 0.960 | – |
| **OpenTime 5B v2** | 42 | 1,568 | 23 | 0 | 2 | **0.958** | 0.962 |
| **OpenTime 5B v2** | 7 | 1,190 | 23 | 0 | 2 | **0.958** | 0.980 |
| **OpenTime 5B v2** | 99 | 1,418 | 21 | 0 | 4 | **0.913** | 0.941 |

- Mean primary F1 across seeds: **0.943** (range 0.913–0.958). Mean F1 of the generated answer: **0.961**.
- Test AUROC of the Yes-vs-No score: 0.998–1.000. Invalid generated answers: **0 of 100** in every seed.
- Paired engine-level bootstrap (10,000 resamples) of the F1 gain over the baseline, 95% intervals: seed 42 [−0.026, +0.186], seed 7 [0.000, +0.163], seed 99 [−0.084, +0.137]. Every lower bound is at or below zero, so **superiority over the baseline is not statistically established** on 100 engines.

## What changed from the pilot

| | Pilot (superseded) | v2 |
|---|---|---|
| Input history | last 30 cycles | the engine's whole past history up to the prediction cycle |
| Channels | 21 sensors + 3 settings | the 14 sensors that vary in FD001 |
| Prompt | fixed question | also states the cycles run so far (known at prediction time) |
| Training data | 16 windows, 1 epoch | 1,190–1,568 windows from 80 engines, 3 epochs |
| Epoch selection | 8 validation windows | lowest validation loss on 96 windows from 20 disjoint engines |
| Decision | greedy text answer | log P(Yes) − log P(No), threshold picked by F1 on 200 truncated histories of the validation engines |

The checkpoint rescales every series to mean 0 and standard deviation 1. With a 30-cycle window that hides the slow drift that signals wear; the full history keeps it visible.

## Protocol and leakage controls

- Engine-disjoint split of the NASA training file: 80 development-train engines, 20 development-validation engines (seeded, same split as the pilot).
- Official test labels were not used for training, epoch selection or the threshold. Each seed was scored on the test set once.
- Every example is past-only: a question at cycle *t* sees rows 1..*t* of that engine (unit-tested in `tests/test_opentime_v2_examples.py`).
- The training script measures its own speed and shrinks the training set (class-balanced) to fit a 40-minute training budget. GPU speed varied between jobs, so the seeds trained on **1,568, 1,190 and 1,418** examples. This is disclosed, not tuned.

## Reproducible evidence

- Model: [`opentime/tslm-nov26`](https://huggingface.co/opentime/tslm-nov26), revision `5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983` (Gemma license; no weights redistributed).
- Reference fine-tuning code: AWS sample `utils.py` pinned to commit `b83a9901d401e9e872ee9048fe61e6c9df9a5b78`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), Nebius `eu-south1` Serverless AI Jobs, PyTorch 2.9.0+cu128, 90-minute cloud timeout per job.
- Seed 42: [GPU run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018761652) (job `aijob-e07zev52b8dwkkyrb6`) and [log recovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021393648).
- Seeds 7 and 99: [GPU run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021536671), artifacts `musitu-nebius-opentime-v2-seed-7` and `-seed-99`.
- Machine-readable: [summary](evidence/opentime-v2/summary.json), per-seed metrics and paired baseline audits in [`evidence/opentime-v2/`](evidence/opentime-v2/).
- Code: [`sentinel/opentime_v2_train.py`](../sentinel/opentime_v2_train.py), [`sentinel/opentime_v2_examples.py`](../sentinel/opentime_v2_examples.py), [`sentinel/opentime_v2_audit.py`](../sentinel/opentime_v2_audit.py), [`scripts/opentime_v2_nebius_job.sh`](../scripts/opentime_v2_nebius_job.sh).

## Limitations

- NASA C-MAPSS is **simulated** engine data. Nothing here is evidence for real aircraft maintenance.
- 100 test engines are few; the bootstrap intervals above do not cover dataset shift.
- Binary 30-cycle warning only; the 5B model does not predict remaining life in cycles, so no MAE comparison.
- The compact temporal Transformer (F1 0.960) is a separate non-OpenTime model. OpenTime v2 is on par with it, not ahead of it.

## Harder test: FD004 (six operating conditions, two fault modes)

Same recipe and checkpoint, plus one change: each sensor is standardised within its operating condition using statistics from the FD004 development-train engines only. 199 train / 50 validation development engines. Test: the 237 official FD004 test engines with at least 30 cycles (11 shorter ones excluded, as in our earlier classical FD004 audit), 53 of them at risk. Each seed scored once.

| System | Caught | False alarms | Missed | Warning F1 |
|---|---:|---:|---:|---:|
| Classical, trained on FD001–FD003 only (never saw FD004) | 40 | 5 | 13 | 0.816 |
| Classical, same FD004 split and normalisation (like-for-like) | 48 | 7 | 5 | 0.889 |
| **OpenTime 5B, seed 42** | 49 | 10 | 4 | **0.875** |
| **OpenTime 5B, seed 7** | 52 | 9 | 1 | **0.912** |
| **OpenTime 5B, seed 99** | 50 | 3 | 3 | **0.943** |

- Mean F1 **0.910**; test AUROC 0.981–0.994; generated answers F1 0.833–0.895 with 0 invalid outputs; 1,406–1,440 training examples per seed.
- Against the like-for-like classical model, OpenTime is ahead in 2 of 3 seeds and behind in 1. Paired bootstrap 95% intervals on the F1 gain: seed 42 [−0.074, +0.044], seed 7 [−0.040, +0.088], seed 99 [+0.009, +0.108].
- OpenTime catches more at-risk engines than the like-for-like classical model in every seed (49–52 vs 48), at the cost of more false alarms in two seeds.
- The like-for-like classical baseline (`sentinel/fd004_classical_same_split.py`) was built after the first FD004 seed finished, with settings copied unchanged from `multidomain_classifier.py`.
- Evidence: [GPU runs, all three seeds](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45052882650) (Nebius jobs `aijob-e07zp93htraq7xbvx3`, `aijob-e07tmgqm7ycnphfy1r`, `aijob-e07nw1dn37mka833ta`), [summary](evidence/opentime-fd004/summary.json), [per-seed metrics and classical baseline](evidence/opentime-fd004/).

---

## Superseded: first pilot (16 examples, one epoch)

Kept for the record. Same checkpoint and GPU; Nebius job `aijob-e07ag71cvmbtjv2s7w`, 04:07:55–04:13:37 UTC, script runtime 321.94 s. Trained on 16 balanced 30-cycle windows, validated on 8, scored on the same 100 official test engines.

| Metric | Before adaptation | After 1 epoch |
| --- | ---: | ---: |
| 30-cycle warning accuracy | 24% | 75% |
| Warning precision | 0.000 | 0.500 |
| Warning recall | 0.000 | 0.240 |
| Warning F1 | 0.000 | 0.324 |
| Invalid model answers | 64 / 100 | 0 / 100 |

Evidence: [GPU run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462), [log recovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004), [metrics](evidence/opentime-nov26-fd001-gpu-pilot-metrics.json), [runner](../sentinel/opentime_train.py).
