# TimeNet Round 1 — MUSITU AXIOM Sentinel final submission

**Project:** MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence

**Team:** MUSITU AXIOM (Evans Musitu; Tally Vans)

**Problem:** Operators need early warning that an engine is approaching end of useful life, with a verifiable understanding of which time-series observations were available at the prediction moment. We evaluate a transparent maintenance-warning prototype on NASA's *simulated* turbofan trajectories—not on live aircraft or real maintenance cases.

## Official deliverable links (verified)

- **TimeNet dataset connector pull request:** https://github.com/OpenTSLM/TimeNet/pull/836 — **open and ready for upstream review**, not merged; full dataset conversion passed in CI; upstream review remains open.
- **Project GitHub repo:** https://github.com/evansmusitu/musitu-timenet-sentinel
- **Live interactive demo:** https://evansmusitu.github.io/musitu-timenet-sentinel/
- **NASA FD001 reproducible classical benchmark:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482
- **Original compact temporal QA Transformer training run:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425
- **OpenTime 5B v2 fine-tune (current result):** [seed 42 GPU run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018761652), [seeds 7 and 99](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021536671), [evidence summary](evidence/opentime-v2/summary.json), [full report](OPENTIME_5B_VERIFIED_GPU_RESULTS.md)
- Superseded first pilot: https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462

## What we built for TimeNet

1. NASA Ames C-MAPSS FD001 connector for engine-disjoint TRAIN / VALIDATION and official TEST engine grouping. 30-cycle windows, three operating settings and 21 sensor streams. NASA source archive is downloaded on demand and checksum-verified. No raw source dataset is redistributed in the PR.
2. A browser demo for explaining maintenance warning scores, uncertainty and operational utility.
3. Two fully reproducible non-OpenTime baselines to establish honest evaluation anchors.
4. A full fine-tune of the organizer's `opentime/tslm-nov26` (5.24B) on Nebius RTX PRO 6000 GPUs: all 80 development-train engines, full past-only histories, 14 varying sensors and the cycle count in the prompt, three training seeds. Epoch and threshold were chosen on 20 disjoint validation engines; each seed was scored once on the 100 official test engines. The [full GPU evidence](OPENTIME_5B_VERIFIED_GPU_RESULTS.md) includes model SHA, outcomes and limitations. (A first 16-example pilot scored F1 0.324 and is superseded.)

## Held-out NASA official test evidence

| Model | Test set | RUL MAE | RUL RMSE | 30-cycle warning F1 |
| --- | ---: | ---: | ---: | ---: |
| Classical gradient-boosted baseline | 100 engine endpoints | 19.4025 cycles | 27.2319 cycles | 0.8889 |
| Separate compact temporal QA Transformer | 100 engine endpoints | 19.0436 cycles | 26.4463 cycles | 0.9600 |
| **OpenTime TSLM November 2026 5B v2, 3 seeds** | 100 engine endpoints | Not measured (binary QA) | Not measured | **0.958 / 0.958 / 0.913** (mean 0.943) |
| OpenTime 5B first pilot (superseded) | 100 engine endpoints | Not measured | Not measured | 0.3243 |

OpenTime v2 beats the classical baseline in all three seeds (20 → 21–23 of 25 at-risk engines caught, zero false alarms) and is on par with the separate compact temporal QA Transformer, which is **not** an OpenTime model. Paired bootstrap 95% intervals on the F1 gain over the baseline have lower bounds of −0.084 to 0.000, so superiority is not statistically established on 100 engines.

**Practical impact:** demonstrate the workflow for detecting end-of-life risk in multivariate telemetry with a clean provenance trail and no training/test engine overlap. All evidence derives from simulation; not validated for real aviation maintenance.

## OpenTime evidence and limitations

Each v2 seed ran as one Nebius Serverless AI Job on a 96 GB RTX PRO 6000 with a 90-minute cloud timeout and automatic cancellation on failure, using an immutable checkpoint revision. The seeds trained on 1,568, 1,190 and 1,418 examples, because the script fits the training set to a 40-minute budget from its measured speed. The model gives a binary 30-cycle warning, not remaining life in cycles. The Gemma model license applies to the checkpoint; no weights are redistributed. All data is NASA simulation.

## Public outreach links required by the organizer

**X post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post. Do not substitute an unrelated earlier project post.

**LinkedIn post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post.

**Discord OpenTime showcase message URL:** To be added after posting in the organizer Discord.

The official TimeNet hackathon requires public X and LinkedIn URLs and a Discord project post. **This document alone is not a complete official submission.** The application also requires the separate project submission form to be completed at https://tally.so/r/J1qbyK by **November 22, 2026 at 11:59 PM PST (November 23, 2026 at 09:59 Africa/Harare)**.

## Submission-ready short pitch

**MUSITU AXIOM Sentinel** warns maintenance teams that an engine is within 30 cycles of end of life, using only telemetry available at that moment. Our TimeNet connector turns NASA's simulated C-MAPSS turbofan histories into engine-disjoint, leak-aware datasets. We fine-tuned the organizer's 5B OpenTime time-series language model on full engine histories: on 100 held-out NASA test engines it scores warning F1 0.913–0.958 across three seeds with zero false alarms, beating our classical baseline (0.889) in every seed and matching our best specialised model (0.960). Every number comes from public CI runs, with confidence intervals and limitations stated. Demo, code, GPU logs and the TimeNet PR are public.

## Truthfulness checks

- Claim the v2 results as measured (F1 0.913–0.958 across three seeds). Say "beats the classical baseline in every seed"; do not say "statistically significant" or "proven superior", and do not claim it beats the compact temporal Transformer.
- Do not claim NASA's dataset has an SPDX license not verified from its record.
- Do not claim PR #836 is merged or approved; it is open and ready for review but not merged.
- Do not claim a Tally submission receipt until the form confirms it.


## Copy-ready organizer social posts

See [SOCIAL_POSTS_FINAL.md](SOCIAL_POSTS_FINAL.md) for the current X, LinkedIn and Discord posts and the screenshot list. The earlier drafts that quoted the 0.324 pilot are retired.
