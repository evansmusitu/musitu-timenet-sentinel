# TimeNet Round 1 — 90-second judges' walkthrough

**MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence**  
Project: https://evansmusitu.github.io/musitu-timenet-sentinel/  
TimeNet connector: https://github.com/OpenTSLM/TimeNet/pull/836 (open for review; **not merged**)

## The 90-second demonstration

**0–15s — The problem.** "Imagine a maintenance analyst looking at a complex machine with 21 sensors and three operating settings. Can we identify an early warning without looking into its future? We use NASA C-MAPSS FD001 simulated engine histories to test that question. This is a research benchmark—not a deployed aviation system."

**15–35s — The actual demo.** Open the [interactive Sentinel dashboard](https://evansmusitu.github.io/musitu-timenet-sentinel/#demo). Select **Engine 041**. It has **18 cycles of actual remaining life**. The independent classical baseline predicts **38.68 cycles**, missing the warning threshold; our separately trained compact temporal QA model predicts **28.42 cycles** and its warning head flags it. Switch models to see the two results. This is **one selected held-out example, not representative of all cases**; the complete evaluations use 100 official test engines.

**35–55s — TimeNet contribution.** Point to [upstream PR #836](https://github.com/OpenTSLM/TimeNet/pull/836). It adds a real C-MAPSS FD001 dataset connector with **30-cycle past-only windows**, **engine-disjoint development splits**, separate official held-out test engines, source checksums and no raw-data redistribution. The connector conversion has been [exercised in CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984278576), while upstream review and license provenance remain unresolved.

**55–75s — Honest OpenTime experiment.** We loaded the organizer's actual **5.24B-parameter OpenTime November 2026 checkpoint** on one **Nebius RTX PRO 6000 Blackwell (96 GB)**, trained for **one epoch on 16 windows** with **eight validation windows**, and evaluated against **100 held-out official test engines**. The binary 30-cycle warning F1 improved from **0.0000 to 0.3243**; invalid answers fell **64 to zero**; warning recall after the run was only **0.24**. This is encouraging format adaptation, **not a competitive risk detector yet**. [Verified log and GPU evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

**75–90s — What next.** "What makes Sentinel valuable is its auditable path from time series and source provenance to leak-aware datasets, actual evaluation, and a human-inspectable result. Our separate CPU models are stronger today; the genuine OpenTime pilot shows precisely what still needs work. We have not claimed deployment or safety certification."

## Reproducible scorecard — official 100-engine FD001 endpoints

| Measured system | RUL MAE (cycles) | 30-cycle warning F1 | Relationship to OpenTime 5B |
|---|---:|---:|---|
| Classical gradient-boosted regression + warning threshold | 19.4025 | 0.8889 | Independent non-OpenTime baseline |
| Compact trained temporal QA Transformer | 19.0436 | 0.9600 | Separate original architecture, **not** organizer checkpoint |
| Genuine OpenTime November 2026 5B **after** small adaptation pilot | Not measured (binary QA only) | **0.3243** (before: 0.0000) | Actual organizer checkpoint, one epoch/16 training examples |

### Evidence in one click

- [Project source and reproducibility](https://github.com/evansmusitu/musitu-timenet-sentinel)
- [Live demo](https://evansmusitu.github.io/musitu-timenet-sentinel/)
- [Real TimeNet dataset contribution PR #836](https://github.com/OpenTSLM/TimeNet/pull/836)
- [Successful classical benchmark](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482)
- [Successful compact temporal training and benchmark](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425)
- [Original successful Nebius 5B job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462)
- [Verified GPU log recovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004)
- [Machine-readable OpenTime evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/opentime-nov26-fd001-gpu-pilot-metrics.json)

### Boundaries of the claim

*NASA FD001 is simulated data, not live aircraft or real maintenance telemetry.* The demo displays selected held-out predictions from independent CPU-trained models; it does **not** execute the 5B checkpoint live. The OpenTime pilot predicts a binary warning response (no RUL MAE). Pilot size is too small for operational conclusions. No source NASA archive or licensed model weights are redistributed. The dataset license needs upstream clarification. PR #836 is still **open**, not merged.

**Competition submission status:** a GitHub demo or PR alone is not an official entry. The real submission requires public X and LinkedIn project posts, a Discord #showcase post, and confirmation from the official Tally project form. Do not claim these steps complete without published URLs and a receipt.
