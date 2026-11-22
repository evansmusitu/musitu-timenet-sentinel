# TimeNet Round 1 — MUSITU AXIOM Sentinel final submission

**Project:** MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence

**Team:** MUSITU AXIOM (Evans Musitu; Tally Vans)

**Problem:** Operators need early warning that an engine is approaching end of useful life, with a verifiable understanding of which time-series observations were available at the prediction moment. We evaluate a transparent maintenance-warning prototype on NASA's *simulated* turbofan trajectories—not on live aircraft or real maintenance cases.

## Official deliverable links (verified)

- **TimeNet dataset connector pull request:** https://github.com/OpenTSLM/TimeNet/pull/836 — **open and ready for upstream review**, not merged; full builder and metadata review remain open.
- **Project GitHub repo:** https://github.com/evansmusitu/musitu-timenet-sentinel
- **Live interactive demo:** https://evansmusitu.github.io/musitu-timenet-sentinel/
- **NASA FD001 reproducible classical benchmark:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482
- **Original compact temporal QA Transformer training run:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425
- **Authentic OpenTime 5B Nebius GPU pilot (live run; outcome pending):** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462

## What we built for TimeNet

1. NASA Ames C-MAPSS FD001 connector for engine-disjoint TRAIN / VALIDATION and official TEST engine grouping. 30-cycle windows, three operating settings and 21 sensor streams. NASA source archive is downloaded on demand and checksum-verified. No raw source dataset is redistributed in the PR.
2. A browser demo for explaining maintenance warning scores, uncertainty and operational utility.
3. Two fully reproducible non-OpenTime baselines to establish honest evaluation anchors.
4. GPU orchestration and a one-hour-bounded Nebius Serverless AI Job to test the organizer-shared `opentime/tslm-nov26` checkpoint. **Do not call the 5B experiment successful until the GPU job logs and metrics are inspected.**

## Held-out NASA official test evidence

| Model | Test set | RUL MAE | RUL RMSE | 30-cycle warning F1 |
| --- | ---: | ---: | ---: | ---: |
| Classical gradient-boosted baseline | 100 engine endpoints | 19.4025 cycles | 27.2319 cycles | 0.8889 |
| Separate compact temporal QA Transformer | 100 engine endpoints | 19.0436 cycles | 26.4463 cycles | 0.9600 |
| OpenTime TSLM November 2026 5B checkpoint | **Pending live GPU result** | — | — | — |

The temporal QA model is an original compact fixed-question architecture **not** the organizer's 5B model. The two rows use separate trained systems and cannot imply OpenTime beat the baselines.

**Practical impact:** demonstrate the workflow for detecting end-of-life risk in multivariate telemetry with a clean provenance trail and no training/test engine overlap. All evidence derives from simulation; not validated for real aviation maintenance.

## OpenTime pilot evidence and limitations

The 5B job uses a single 96 GiB NVIDIA RTX PRO 6000 GPU, an immutable Hugging Face model revision, past-only 24-channel observations and a small training/validation pilot. Training results and weights are not yet established. Nebius cloud-side runtime timeout is 1 hour; the independent watchdog and exact GitHub workflow are linked in the repo. The Gemma model license applies to the checkpoint; no weights are redistributed.

## Public outreach links required by the organizer

**X post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post. Do not substitute an unrelated earlier project post.

**LinkedIn post URL:** To be added after publishing an authentic MUSITU Sentinel-specific public post.

**Discord OpenTime showcase message URL:** To be added after posting in the organizer Discord.

The official TimeNet hackathon requires public X and LinkedIn URLs and a Discord project post. **This document alone is not a complete official submission.** The application also requires the separate project submission form to be completed at https://tally.so/r/J1qbyK by **November 22, 2026 at 11:59 PM PST (November 23, 2026 at 09:59 Africa/Harare)**.

## Submission-ready short pitch

**MUSITU AXIOM Sentinel** makes the cost of failure understandable before maintenance teams run out of time. Using NASA Ames simulated turbofan lifecycles, our TimeNet connector assembles leak-aware 30-cycle multivariate windows and official held-out engine endpoints. A reproducible classical benchmark reaches 19.40-cycle remaining-useful-life MAE and 0.889 F1 for warnings within 30 cycles. A separate compact trained temporal QA Transformer reaches 19.04-cycle MAE and 0.960 warning F1. Our interactive browser demo explains the decision boundary, dataset provenance and operational limitations. We are also testing the official OpenTime November 2026 5B time-series language model on a constrained Nebius GPU job, with its results kept separate from verified baselines until available. Source, CI evidence, demo and a TimeNet connector PR are public.

## Truthfulness checks

- Do not claim OpenTime 5B training or test success without a completed logged experiment.
- Do not claim NASA's dataset has an SPDX license not verified from its record.
- Do not claim PR #836 is merged or approved; it remains an open draft.
- Do not claim a Tally submission receipt until the form confirms it.


## Copy-ready organizer social posts

**X / Twitter — under 280 characters:**

🚀 MUSITU AXIOM Sentinel is live for #TimeNet! Predictive-maintenance AI on NASA simulated engines, with reproducible results from 100 held-out engines. Try our demo: https://evansmusitu.github.io/musitu-timenet-sentinel/ #OpenTime

**LinkedIn:**

We built MUSITU AXIOM Sentinel for the TimeNet / OpenTime Hackathon.

Using NASA C-MAPSS simulated engine histories, our TimeNet connector assembles leak-aware multivariate windows with 21 sensors and three operating settings. A reproducible classical baseline achieves 19.40-cycle remaining-useful-life MAE and F1 0.889 for early warnings on 100 official held-out engines. A separate compact temporal QA Transformer reaches F1 0.960 (not the OpenTime 5B model).

Our actual OpenTime 5B GPU experiment is running on Nebius; its results are not yet verified.

🌐 Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/
🧪 Code: https://github.com/evansmusitu/musitu-timenet-sentinel
🔀 TimeNet PR: https://github.com/OpenTSLM/TimeNet/pull/836

#TimeNet #OpenTime #PredictiveMaintenance #AI #MUSITU

**Discord showcase:**

MUSITU AXIOM Sentinel — TimeNet Hackathon. We've built a NASA C-MAPSS FD001 leak-aware connector, a live demo and reproducible 100-engine predictive-maintenance benchmarks. The organizer OpenTime 5B pilot is running on a time-limited Nebius RTX PRO 6000 and we will publish its verified outcomes separately. Feedback welcome! Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/ | Repo: https://github.com/evansmusitu/musitu-timenet-sentinel | PR: https://github.com/OpenTSLM/TimeNet/pull/836.
