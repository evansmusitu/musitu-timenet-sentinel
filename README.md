# MUSITU AXIOM Sentinel — TimeNet / OpenTime Hackathon

Evidence-first predictive maintenance prototype using NASA Ames C-MAPSS FD001 simulated turbofan engine telemetry.

**Status:** NASA C-MAPSS FD001 CPU benchmarks reproduced and OpenTime November 2026 5B checkpoint **genuinely loaded, fine-tuned for one small epoch and evaluated** on an RTX PRO 6000 GPU. 100-engine held-out warning F1 increased from 0.000 to 0.324 (accuracy 24% to 75%); still below the separate CPU baselines. See [verified GPU evidence](docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

## TimeNet Round 1 evidence package

- **Next OpenTime training protocol (CPU-prepared, not run on GPU):** [Engine-diverse v2 sampling manifest](docs/evidence/opentime_v2_sampling_preflight.json) · [successful source-data preflight](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45019219005). The proposed 256 development-training windows cover 80 engines with a separate 64-window validation set across 20 other engines; this is **not** a new 5B score or a guaranteed improvement. The original pilot was 16 training windows and remains unchanged.
- **Measured sequence dependence (new, CPU-only):** [Frozen compact temporal QA ablation results](docs/evidence/frozen_sequence_integrity.json) · [reproducible CI run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018569313). Across 100 held-out simulated FD001 engines, warning F1 falls from 0.960 (original) to 0.684 (reversed time), 0.844 (shuffled time), and 0.387 (mean-only history). Exploratory post-hoc controls on the **separate compact model**, **not** OpenTime 5B or causal explanation verification.
- **90-second judges' walkthrough:** [Concrete engine demo, evidence links, comparison and limitations](docs/TIMENET_ROUND1_JUDGES_GUIDE.md).
- **Chronological maintenance decision replay:** [2,163 authentic past-only recorded test windows, 24/25 eligible engines warned, and eight premature-alert engines](docs/maintenance-replay.html). **Frozen independent compact model, not OpenTime 5B**; overlapping windows are correlated, 75 test engines never entered the observable final 30-cycle horizon. [Full CPU Actions proof](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45037237604).
- **Verified 100-engine decision audit:** [Confusion matrices, uncertainty intervals and an interactive maintenance-warning trade-off](docs/decision-audit.html).
- **Cross-cohort robustness lab (new):** [Independent FD001-frozen model transfer across four NASA simulated cohorts, and exploratory FD004-heldout multi-domain models](docs/robustness-lab.html). Experiments use CPU-only models **not OpenTime 5B**.
- **Competition submission pack:** [Technical pitch, TimeNet PR, evaluation evidence and limitations](docs/TIMENET_ROUND1_FINAL_SUBMISSION.md).
- **Completed organizer OpenTime 5B pilot:** [Verified model metrics and limitations](docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md) · [successful Nebius run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462) · [recovered job logs](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004).
- **Real-time GPU capacity:** [Nebius RTX 6000 96 GiB quota-clipped availability](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44982965639).
- **Independent cloud safety monitor:** [Read-only verification](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983644358) and time-limited native Nebius Job configuration.
- **Round 1 entry:** the team owner reports that the official [Tally project form](https://tally.so/r/J1qbyK) was submitted November 22, 2026. An independent organizer receipt has not been verified. Do not submit a duplicate entry.

## Live evidence

- **Interactive demonstration:** https://evansmusitu.github.io/musitu-timenet-sentinel/
- **Successful official NASA FD001 CPU benchmark:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482
- **Download artifact:** open that run's Artifacts section and download `nasa-cmapss-results`.
- **Decision-grade audit of both original 100-engine CSVs:** baseline TP20/FP0/FN5/TN75 versus separate compact temporal head TP24/FP1/FN1/TN74. The compact head caught four additional at-risk simulated engines with one additional false alarm. [Interactive dashboard](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html) · [reproducible aggregate evidence](docs/evidence/heldout_decision_audit.json). Paired 95% bootstrap confidence intervals cross zero; superiority is not proven.
- **Official test on 100 unseen engines:** 19.4025-cycle RUL MAE; 27.2319-cycle RMSE; 30-cycle warning F1 0.8889, precision 1.0, recall 0.8.
- **Separate experimental fixed-prompt temporal QA Transformer:** source `sentinel/temporal_qa.py`, workflow `.github/workflows/temporal.yml`. The 12-epoch training run completed successfully: 19.0436-cycle RUL MAE, 26.4463-cycle RMSE, and F1 0.960 from its trained 30-cycle warning head on 100 official test engines. [Check training and model artifact](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425). This is **not** OpenTime's 5B model or a general-purpose TSLM.

## Official TimeNet connector contribution

- **Upstream PR ready for review:** https://github.com/OpenTSLM/TimeNet/pull/836
- Fork branch: https://github.com/evansmusitu/TimeNet/tree/feat/musitu-cmapss-fd001
- PR is open and ready for upstream review, not merged. Added [11 synthetic fixture tests and strict source-data validation](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45002805232), verified passing in TimeNet's actual locked environment alongside real-source NASA conversion. Maintainer review and source-data license clarification remain open. No raw data redistributed.

## Exploratory cross-domain simulated warning research (separate CPU-only models)

This additional research exposed meaningful domain shift rather than hiding negative results. Frozen FD001-only models were tested without retraining against FD002, FD003 and FD004 official NASA **simulated** engine cohorts. The original compact temporal model's warning F1 on FD004 was 0.479 with 112 false alarms among 237 eligible engines; the classical FD001-only warning F1 was 0.000. The evidence warns against deploying FD001-only models on unfamiliar operating conditions.

We then explored separate CPU classical models trained on **FD001/FD002/FD003 development engines**, with FD004 held entirely out of fitting and threshold selection. The gradient-boosted regression warning F1 on eligible official FD004 engines was **0.643**. A direct-warning classifier selected its probability threshold 0.40 using only independent development validation engine groups and obtained **F1 0.8163** (40 true warnings, 5 false alarms, 13 misses, 179 true nonwarnings) on the same **237 eligible FD004 engines**, excluding 11 test trajectories shorter than the required 30 cycles. This is a **post-hoc, exploratory simulation experiment, not a preregistered or operational safety study**. It is **not** the organizer's 5B checkpoint or evidence of cross-manufacturer transfer.

[Open the public, aggregate-only robustness lab](https://evansmusitu.github.io/musitu-timenet-sentinel/robustness-lab.html) · [Original classical transfer CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003875499) · [Original compact checkpoint transfer CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003991598) · [Multi-domain classifier CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45004601674).

## Experimental OpenTime checkpoint adaptation (small GPU pilot completed)

A new GPU runner is checked in at `sentinel/opentime_train.py`. It uses the organizer-recommended AWS reference implementation pinned to commit `b83a9901d401e9e872ee9048fe61e6c9df9a5b78` and explicit immutable HF checkpoint revision. It constructs past-only 24-channel windows, freezes the base model according to the upstream recipe, selects the best epoch on development engines only, and logs before/after performance and raw predictions on the fixed 100-engine official test set. The model was loaded on a genuine RTX PRO 6000 GPU and completed one epoch, with F1=0.324 on 100 held-out NASA official test engines. The data adapter was also exercised with regression tests. See [full reproducible GPU report](docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

```bash
# Run on CUDA-enabled PyTorch 2.9 GPU image; recommend one 80 GB GPU.
python -m pip install -r requirements-opentime.txt
REV=$(python -c 'from huggingface_hub import model_info; print(model_info("opentime/tslm-nov26").sha)')
python -m sentinel.opentime_train --revision "$REV" --epochs 1 --train-limit 48 --val-limit 24
```

Results and adapted parameter weights go to `artifacts/opentime/`. Inspect the checkpoint's Gemma license prior to use or distribution. No GPU execution or paid cloud provisioning is performed by this script.

## Reproduce

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python -m sentinel.benchmark --download
```

The runner fetches the NASA FD001 archive from the versioned Zenodo record and verifies its MD5, then produces `artifacts/metrics.json`, `artifacts/official_test_predictions.csv` and `artifacts/summary.md`.

**Method:** 30-cycle past-only windows, 21 sensor streams, 3 operating settings, engine-group-disjoint development split, held-out NASA test engines, gradient-boosted regression of remaining useful life (RUL) plus 30-cycle warning classification. Baseline is classical machine learning, not the OpenTime TSLM.

The experimental checkpoint shared in OpenTime Discord is https://huggingface.co/opentime/tslm-nov26 . Martin noted that novel model components are not yet included in the main OpenTSLM repository; inspect the example helpers at https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/examples/opentslm-finetune-arfbench/scripts/utils.py .

## Source

NASA Ames Prognostics Center of Excellence: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Archived source: https://zenodo.org/records/15346912 (MD5 `79a22f36e80606c69d0e9e4da5bb2b7a`). We do **not** redistribute source data.

TimeNet: https://github.com/OpenTSLM/TimeNet . Current challenge: https://challenge.timenet.ai/ . Submission deadline 2026-11-22 11:59 p.m. PST, official form https://tally.so/r/J1qbyK .

**Limitations:** Engine data is simulated. Do not use these predictions for real aircraft maintenance. Results from a CPU baseline must not be presented as TSLM results. The upstream PR is **open and ready for review, not merged**, and the demo is published. Remaining: upstream TimeNet maintainer review and dataset-license clarification; independent verification of the user's reported Tally submission and public social-post permalinks. The OpenTime GPU pilot has been completed and verified but needs larger training experiments to improve warning F1.
