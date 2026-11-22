# MUSITU AXIOM Sentinel — TimeNet / OpenTime Hackathon

Evidence-first predictive maintenance prototype using NASA Ames C-MAPSS FD001 simulated turbofan engine telemetry.

**Status:** the organizer's OpenTime November 2026 5B time-series language model, fine-tuned on all 80 FD001 development-train engines, reaches **30-cycle warning F1 0.958 / 0.958 / 0.913** across three training seeds on the 100 official NASA test engines, with zero false alarms. That beats our classical gradient-boosting baseline (0.889) in every seed and is on par with our separate compact temporal Transformer (0.960). Bootstrap intervals on the gain touch zero, so superiority is not statistically established. The first 16-example pilot (F1 0.324) is superseded. See [verified GPU evidence](docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

| 100 official FD001 test engines (25 at risk) | Caught | False alarms | Missed | Warning F1 |
|---|---:|---:|---:|---:|
| Classical gradient boosting | 20 | 0 | 5 | 0.889 |
| Compact temporal QA Transformer (not OpenTime) | 24 | 1 | 1 | 0.960 |
| **OpenTime 5B v2, seed 42 / 7 / 99** | 23 / 23 / 21 | 0 / 0 / 0 | 2 / 2 / 4 | **0.958 / 0.958 / 0.913** |

## TimeNet Round 1 evidence package

- **90-second judges' walkthrough:** [Concrete engine demo, evidence links, comparison and limitations](docs/TIMENET_ROUND1_JUDGES_GUIDE.md).
- **Verified 100-engine decision audit:** [Confusion matrices, uncertainty intervals and an interactive maintenance-warning trade-off](docs/decision-audit.html).
- **Cross-cohort robustness lab (new):** [Independent FD001-frozen model transfer across four NASA simulated cohorts, and exploratory FD004-heldout multi-domain models](docs/robustness-lab.html). Experiments use CPU-only models **not OpenTime 5B**.
- **Competition submission pack:** [Technical pitch, TimeNet PR, evaluation evidence and limitations](docs/TIMENET_ROUND1_FINAL_SUBMISSION.md).
- **OpenTime 5B v2 fine-tune:** [Verified metrics and limitations](docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md) · [seed 42 Nebius run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018761652) · [seeds 7 and 99](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021536671) · [evidence summary](docs/evidence/opentime-v2/summary.json). Superseded pilot: [run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462).
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

## OpenTime 5B fine-tune

`sentinel/opentime_v2_train.py` fine-tunes `opentime/tslm-nov26` with the organizer-recommended AWS reference implementation (pinned to commit `b83a9901d401e9e872ee9048fe61e6c9df9a5b78`) and an immutable checkpoint revision. It gives the model each engine's full past-only history of the 14 varying sensors plus the cycle count, trains on the 80 development-train engines, selects the epoch by validation loss on 20 disjoint engines, picks the warning threshold on truncated validation histories, and then scores the 100 official test engines once. `sentinel/opentime_v2_audit.py` compares the result with the classical baseline engine by engine. Nebius jobs: `.github/workflows/nebius-opentime-v2.yml` and `nebius-opentime-v2-seeds.yml`.

```bash
# Run on a CUDA PyTorch 2.9 image with one GPU of at least 48 GB.
python -m pip install -r requirements-opentime.txt
python -m sentinel.opentime_v2_train --revision 5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983 --seed 42 --epochs 3
```

Results go to `artifacts/opentime_v2/`. The first pilot's runner (`sentinel/opentime_train.py`) is kept for reproducibility. Inspect the checkpoint's Gemma license prior to use or distribution. No GPU execution or paid cloud provisioning is performed by this script.

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

**Limitations:** Engine data is simulated. Do not use these predictions for real aircraft maintenance. Results from the CPU baselines must not be presented as TSLM results; the OpenTime v2 numbers are binary 30-cycle warnings on 100 test engines, not remaining-life estimates. The upstream PR is **open and ready for review, not merged**, and the demo is published. Remaining: upstream TimeNet maintainer review and dataset-license clarification; independent verification of the user's reported Tally submission and public social-post permalinks. OpenTime v2 seeds trained on 1,568, 1,190 and 1,418 examples because the script fits the training set to a time budget.
