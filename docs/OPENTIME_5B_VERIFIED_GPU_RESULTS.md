# Verified OpenTime November 2026 5B GPU pilot — NASA FD001

**Result:** real OpenTime TSLM checkpoint loaded, fine-tuned for one epoch on a small NASA C-MAPSS FD001 training pilot, and evaluated before/after on 100 official held-out engine endpoints. The Nebius AI Job and GitHub Actions both ended successfully.

## Reproducible evidence

- Exact model: [`opentime/tslm-nov26`](https://huggingface.co/opentime/tslm-nov26), pinned Hugging Face revision `5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983` (Gemma license).
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
- Platform: Nebius `eu-south1` Serverless AI Job `aijob-e07ag71cvmbtjv2s7w`; PyTorch `2.9.0+cu128`.
- Started: **2026-11-22 04:07:55 UTC**; completed **04:13:37 UTC**. Model-run script reported **321.94 seconds**.
- **Training:** 16 selected training windows from 80 development engines; **validation:** 8 selected windows from 20 disjoint development engines; one epoch.
- **Test:** 100 official held-out NASA FD001 engines, each with its final provided observation history; no official test labels used for training or epoch selection.
- [Successful Nebius GPU GitHub Actions run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462)
- [Independent log-recovery and metric validation run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004); contains the original Nebius job logs, final state and machine-readable metrics as a downloadable `musitu-opentime-5b-gpu-recovered-logs` artifact.
- [Machine-readable reproduced metrics](evidence/opentime-nov26-fd001-gpu-pilot-metrics.json)
- [GPU runner](../sentinel/opentime_train.py) and [Nebius job container entrypoint](../scripts/opentime_nebius_job.sh).

## Before / after on official test engines

| Metric | Before adaptation | After 1 epoch |
| --- | ---: | ---: |
| 30-cycle warning accuracy | 24% | 75% |
| Warning precision | 0.000 | 0.500 |
| Warning recall | 0.000 | 0.240 |
| Warning F1 | 0.000 | **0.324** |
| Balanced accuracy (invalid as errors) | 0.160 | 0.580 |
| Invalid model answers | 64 / 100 | **0 / 100** |

These are real results for a very small pilot; the improvement in accuracy and valid answer formatting is encouraging, but **warning recall and F1 remain poor**. Do not present the result as outperforming our separate CPU baselines or proving operational readiness.

The classical gradient-boosted NASA FD001 baseline reaches **19.40-cycle RUL MAE** and **0.889 F1**, and a separate compact temporal QA Transformer reaches **19.04-cycle RUL MAE** and **0.960 F1**. The latter is **not** the OpenTime 5B model. The 5B test reports **binary Yes/No warning QA**, not RUL regression MAE, so numeric MAE comparisons are not applicable.

## Limitations

- Simulated engine telemetry only, **not real aviation maintenance validation**.
- Tiny pilot: 16 training examples, eight validation examples, one epoch.
- No full-dataset fine-tuning or hyperparameter search was conducted.
- A later larger experiment would require a new budget and independent capacity/billing review.
- Checkpoint's Gemma licensing remains applicable; no checkpoint weights are redistributed here.
- GPU job completed automatically; no GPU VM remains scheduled from this one-shot task. Additional temporary storage billing must be checked separately.
