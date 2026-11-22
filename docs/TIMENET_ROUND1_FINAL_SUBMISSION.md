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
- **Verified real OpenTime 5B GPU pilot:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462 — [recovered logs and machine-readable before/after metrics](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004)

## What we built for TimeNet

1. NASA Ames C-MAPSS FD001 connector for engine-disjoint TRAIN / VALIDATION and official TEST engine grouping. 30-cycle windows, three operating settings and 21 sensor streams. NASA source archive is downloaded on demand and checksum-verified. No raw source dataset is redistributed in the PR.
2. A browser demo for explaining maintenance warning scores, uncertainty and operational utility.
3. Two fully reproducible non-OpenTime baselines to establish honest evaluation anchors.
4. A real, time-limited 96 GiB Nebius RTX PRO 6000 GPU run that loaded and fine-tuned organizer-shared `opentime/tslm-nov26` for one epoch on 16 selected training windows, validated on eight windows, and scored the model before/after on the 100 official NASA test engines. The [full GPU evidence](OPENTIME_5B_VERIFIED_GPU_RESULTS.md) includes model SHA, outcomes and limitations.

## Held-out NASA official test evidence

| Model | Test set | RUL MAE | RUL RMSE | 30-cycle warning F1 |
| --- | ---: | ---: | ---: | ---: |
| Classical gradient-boosted baseline | 100 engine endpoints | 19.4025 cycles | 27.2319 cycles | 0.8889 |
| Separate compact temporal QA Transformer | 100 engine endpoints | 19.0436 cycles | 26.4463 cycles | 0.9600 |
| Real OpenTime TSLM November 2026 5B pilot, 1 epoch | 100 engine endpoints | Not measured (binary QA) | Not measured | **0.3243** (before: 0.0000) |

The temporal QA model is an original compact fixed-question architecture **not** the organizer's 5B model. The two rows use separate trained systems and cannot imply OpenTime beat the baselines.

**Practical impact:** demonstrate the workflow for detecting end-of-life risk in multivariate telemetry with a clean provenance trail and no training/test engine overlap. All evidence derives from simulation; not validated for real aviation maintenance.

## OpenTime pilot evidence and limitations

The 5B job uses a single 96 GiB NVIDIA RTX PRO 6000 GPU, an immutable Hugging Face model revision, past-only 24-channel observations and a small training/validation pilot. The [verified log-recovery run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004) confirms training and test completion: warning F1 improved from 0.0000 to 0.3243, accuracy 24% to 75%, invalid answers 64 to zero. This remains below the classical baseline and is not yet operationally useful. The actual model computation lasted 321.94 seconds; Nebius cloud-side runtime timeout was 1 hour, with an independent monitor. The Gemma model license applies to the checkpoint; no weights are redistributed.

## Public outreach links required by the organizer

**X post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post. Do not substitute an unrelated earlier project post.

**LinkedIn post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post.

**Discord OpenTime showcase message URL:** To be added after posting in the organizer Discord.

The official TimeNet hackathon requires public X and LinkedIn URLs and a Discord project post. **This document alone is not a complete official submission.** The application also requires the separate project submission form to be completed at https://tally.so/r/J1qbyK by **November 22, 2026 at 11:59 PM PST (November 23, 2026 at 09:59 Africa/Harare)**.

## Submission-ready short pitch

**MUSITU AXIOM Sentinel** makes the cost of failure understandable before maintenance teams run out of time. Using NASA Ames simulated turbofan lifecycles, our TimeNet connector assembles leak-aware 30-cycle multivariate windows and official held-out engine endpoints. A reproducible classical benchmark reaches 19.40-cycle remaining-useful-life MAE and 0.889 F1 for warnings within 30 cycles. A separate compact trained temporal QA Transformer reaches 19.04-cycle MAE and 0.960 warning F1. Our interactive browser demo explains the decision boundary, dataset provenance and operational limitations. We also genuinely adapted the official OpenTime November 2026 5B time-series language model for one epoch on a Nebius RTX PRO 6000. On 100 held-out test engines, warning F1 improved from 0.000 to 0.324 and invalid outputs fell from 64 to zero; this small pilot still trails our separate CPU baselines. Source, CI evidence, demo and a TimeNet connector PR are public.

## Truthfulness checks

- Claim only the completed 16-example, one-epoch OpenTime 5B pilot and its actual logged F1 0.324; do not claim superior performance or full-data training.
- Do not claim NASA's dataset has an SPDX license not verified from its record.
- Do not claim PR #836 is merged or approved; it is open and ready for review but not merged.
- Do not claim a Tally submission receipt until the form confirms it.


## Copy-ready organizer social posts

**X / Twitter — under 280 characters:**

🚀 MUSITU AXIOM Sentinel is live for #TimeNet! Predictive-maintenance AI on NASA simulated engines, with reproducible results from 100 held-out engines. Try our demo: https://evansmusitu.github.io/musitu-timenet-sentinel/ #OpenTime

**LinkedIn:**

We built MUSITU AXIOM Sentinel for the TimeNet / OpenTime Hackathon.

Using NASA C-MAPSS simulated engine histories, our TimeNet connector assembles leak-aware multivariate windows with 21 sensors and three operating settings. A reproducible classical baseline achieves 19.40-cycle remaining-useful-life MAE and F1 0.889 for early warnings on 100 official held-out engines. A separate compact temporal QA Transformer reaches F1 0.960 (not the OpenTime 5B model).

Our genuine one-epoch OpenTime 5B GPU pilot completed on Nebius RTX PRO 6000: held-out warning F1 improved from 0 to 0.324, while invalid answers fell from 64 to zero. This is still below our separate classical baseline, but the comparison is fully reproducible: https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md

🌐 Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/
🧪 Code: https://github.com/evansmusitu/musitu-timenet-sentinel
🔀 TimeNet PR: https://github.com/OpenTSLM/TimeNet/pull/836

#TimeNet #OpenTime #PredictiveMaintenance #AI #MUSITU

**Discord showcase:**

MUSITU AXIOM Sentinel — TimeNet Hackathon. We've built a NASA C-MAPSS FD001 leak-aware connector, a live demo and reproducible 100-engine predictive-maintenance benchmarks. The organizer OpenTime 5B pilot completed on Nebius RTX PRO 6000: F1 0.000 -> 0.324 on 100 held-out NASA engines, with model evaluation logs published. We describe the small pilot's limitations transparently. Feedback welcome! Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/ | Repo: https://github.com/evansmusitu/musitu-timenet-sentinel | PR: https://github.com/OpenTSLM/TimeNet/pull/836.
