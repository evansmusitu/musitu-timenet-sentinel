# TimeNet Round 1 — 90-second judges' walkthrough

**MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence**  
Project: https://evansmusitu.github.io/musitu-timenet-sentinel/  
TimeNet connector: https://github.com/OpenTSLM/TimeNet/pull/836 (open for review; **not merged**)


## What the trained compact model uses — controlled sequence ablation (new evidence)

A [CPU-only GitHub Actions experiment](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018569313) reloaded our **pinned independent compact QA model** and first reproduced its original 100-engine predictions from the prior successful run. No retraining, threshold retuning or paid GPU run occurred.

| Recorded-input condition | Warning F1 | Correct risk warnings / 25 | Missed warnings / 25 | Remaining-life MAE |
|---|---:|---:|---:|---:|
| Original 30-cycle history | **0.960** | 24 | 1 | 19.04 cycles |
| Reverse time order | 0.684 | 13 | 12 | 28.00 cycles |
| Shuffle time (same permutation across channels per engine) | 0.844 | 19 | 6 | 24.10 cycles |
| Repeat only the last observed values | 0.833 | 20 | 5 | 23.67 cycles |
| Repeat each channel's mean | 0.387 | 6 | 19 | 37.62 cycles |

This is **evidence of input-sequence sensitivity** on NASA simulated engines. It does **not** validate causal explanations, prove the model is robust to real sensor faults, or refer to the separate organizer OpenTime 5B checkpoint. The ablations are exploratory post-hoc analyses of the held-out set, not fresh independent validation. [Full aggregate metrics and provenance](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/frozen_sequence_integrity.json).

## The 90-second demonstration

**0–15s — The problem.** "Imagine a maintenance analyst looking at a complex machine with 21 sensors and three operating settings. Can we identify an early warning without looking into its future? We use NASA C-MAPSS FD001 simulated engine histories to test that question. This is a research benchmark—not a deployed aviation system."

**15–35s — A decision, not just a metric.** Open the [100-engine decision audit](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html). The independent classical baseline identifies 20 of 25 at-risk engines, missing five, with no false alarms. Our separate compact temporal warning head identifies 24 of 25 at-risk engines, missing one, with one false alarm. Then select **Engine 041** in the [interactive engine demo](https://evansmusitu.github.io/musitu-timenet-sentinel/#demo): true remaining life 18 cycles; classical prediction 38.68 (missed), compact prediction 28.42 (warning). This is one deliberately selected held-out example, not a typical case. The complete evaluation spans 100 engines.

**35–55s — TimeNet contribution.** Point to [upstream PR #836](https://github.com/OpenTSLM/TimeNet/pull/836). It adds a real C-MAPSS FD001 dataset connector with **30-cycle past-only windows**, **engine-disjoint development splits**, separate official held-out test engines, source checksums and no raw-data redistribution. The connector now also has 11 purpose-built synthetic tests, strict source/label validation, and a [successful actual TimeNet environment acceptance job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45002805232) including the real NASA dataset conversion. Upstream PR review and precise source license provenance remain unresolved.

**55–75s — Honest OpenTime experiment.** We loaded the organizer's actual **5.24B-parameter OpenTime November 2026 checkpoint** on one **Nebius RTX PRO 6000 Blackwell (96 GB)**, trained for **one epoch on 16 windows** with **eight validation windows**, and evaluated against **100 held-out official test engines**. The binary 30-cycle warning F1 improved from **0.0000 to 0.3243**; invalid answers fell **64 to zero**; warning recall after the run was only **0.24**. This is encouraging format adaptation, **not a competitive risk detector yet**. [Verified log and GPU evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

**75–90s — What next.** "What makes Sentinel valuable is its auditable path from time series and source provenance to leak-aware datasets, actual evaluation, and a human-inspectable result. Our separate CPU models are stronger today; the genuine OpenTime pilot shows precisely what still needs work. We have not claimed deployment or safety certification."

## What 100 engines really show

| Official held-out result | Independent classical baseline | Separate compact temporal QA warning head |
|---|---:|---:|
| True positives / correct early warnings | 20 | 24 |
| False negatives / missed 30-cycle warnings | 5 | 1 |
| False positives / unnecessary warnings | 0 | 1 |
| True negatives / correctly ruled-out risks | 75 | 74 |
| 30-cycle warning F1 | 0.8889 | 0.9600 |

**Trade-off:** the compact head identified **four more** at-risk simulated engines and produced **one additional false alarm** on this held-out test. Explore alternative hypothetical missed-warning vs false-alarm penalties in the [decision audit dashboard](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html); these are **not money or actual maintenance cost savings**.

**Uncertainty:** 10,000 deterministic paired engine-level bootstrap resamples gave the observed F1 delta +0.0711 with a 95% interval **[-0.041, +0.200]**, and RUL MAE improvement (classical minus compact) +0.359 cycles with interval **[-2.939, +3.623]**. Both intervals include zero: do **not** claim proven general superiority, statistical significance, or real aviation deployment. Evidence is independently auditable from the two successful GitHub Actions prediction artifacts and [machine-readable aggregate audit](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/heldout_decision_audit.json).

## Reproducible scorecard — official 100-engine FD001 endpoints

| Measured system | RUL MAE (cycles) | 30-cycle warning F1 | Relationship to OpenTime 5B |
|---|---:|---:|---|
| Classical gradient-boosted regression + warning threshold | 19.4025 | 0.8889 | Independent non-OpenTime baseline |
| Compact trained temporal QA Transformer | 19.0436 | 0.9600 | Separate original architecture, **not** organizer checkpoint |
| Genuine OpenTime November 2026 5B **after** small adaptation pilot | Not measured (binary QA only) | **0.3243** (before: 0.0000) | Actual organizer checkpoint, one epoch/16 training examples |

## Exploratory simulated cross-domain follow-up — separate from the 5B pilot

[Open the public NASA FD001–FD004 robustness lab](https://evansmusitu.github.io/musitu-timenet-sentinel/robustness-lab.html). It reports failure as well as improvement:

- The original FD001-only classical warning model fell to **F1=0.000 on FD004**, while the independently trained compact temporal warning head achieved **F1=0.479**, but with **112 false alarms and one missed at-risk engine** among 237 eligible test cases. These systems are **not OpenTime 5B**.
- In an explicitly **post-hoc, exploratory** experiment, a separate classical regression model trained on FD001/FD002/FD003 development engines reached **F1=0.643** on held-out FD004. A dedicated classical direct-warning classifier used a 0.40 probability threshold selected **only on engine-disjoint development validation**, and reached **F1=0.8163**, precision 0.8889, recall 0.7547: **40 correct warnings, five false alarms, 13 misses, 179 correctly excluded non-warnings**. FD004 data were not used in model training or threshold selection.
- FD004 has 248 official test engines, of which 237 have the required 30-cycle window; **11 short trajectories were explicitly excluded**. All four subsets are NASA simulations, not truly different physical manufacturers or real aircraft.
- Original proof: [FD001-frozen classical transfer CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003875499), [original compact checkpoint transfer CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003991598), [multi-domain regression CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45004350622) and [validation-selected classifier CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45004601674).

These findings support a **need for validation under distribution shift**, not a claim of actual maintenance safety, commercial savings or superiority over a real-world wind benchmark.

### Evidence in one click

- [Project source and reproducibility](https://github.com/evansmusitu/musitu-timenet-sentinel)
- [Live demo](https://evansmusitu.github.io/musitu-timenet-sentinel/)
- [100-engine held-out decision audit with confidence intervals](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html)
- [Passing 34-test main CPU CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45002652070)
- [Passing 11-test TimeNet connector CI with real NASA conversion](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45002805232)
- [Real TimeNet dataset contribution PR #836](https://github.com/OpenTSLM/TimeNet/pull/836)
- [Successful classical benchmark](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482)
- [Successful compact temporal training and benchmark](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425)
- [Original successful Nebius 5B job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462)
- [Verified GPU log recovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004)
- [Machine-readable OpenTime evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/opentime-nov26-fd001-gpu-pilot-metrics.json)

### Boundaries of the claim

*NASA FD001 is simulated data, not live aircraft or real maintenance telemetry.* The demo displays selected held-out predictions from independent CPU-trained models; it does **not** execute the 5B checkpoint live. The OpenTime pilot predicts a binary warning response (no RUL MAE). Pilot size is too small for operational conclusions. No source NASA archive or licensed model weights are redistributed. The dataset license needs upstream clarification. PR #836 is still **open**, not merged.

**Competition submission status:** the team owner reported submitting the official Tally Round 1 form on November 22, 2026. We have not independently obtained an organizer project receipt or public social post permalinks, so do not imply external receipt verification. The entry should not be resubmitted.


## Additional verified longitudinal evidence — November 22 (post-submission)

Open the [chronological maintenance operator replay](https://evansmusitu.github.io/musitu-timenet-sentinel/maintenance-replay.html) after the [original 100-engine endpoint audit](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html).

The CPU-only [frozen-model longitudinal experiment](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45037237604) reproduced the original compact-model endpoint artifact and then evaluated **2,163 past-only 30-cycle windows** across the official 100 NASA FD001 simulated test engines. At the unmodified 0.5 warning threshold and fixed five-cycle replay cadence:

- **24 of 25 engines whose observed data entered the final 30-cycle risk horizon** received at least one correct within-horizon warning; one did not.
- **Eight of 100 engines** received at least one premature warning with true RUL still above 30 cycles.
- Across the correlated test windows: TP 74 / FP 11 / FN 14 / TN 2,064; descriptive window-level F1 **0.8555**. These overlapping windows are **not independent test engines**.
- **75 engines did not have recorded observations within the last 30 cycles before failure**; they cannot be scored as successful or unsuccessful near-failure detections.

This is the **same independently trained compact temporal QA Transformer**, **not** the genuine organizer OpenTime 5B model, which remained F1 **0.3243** after its 16-window GPU pilot. The replay is research-only, with no commercial uptime/savings figures, no unrecorded operational data, and no revised Tally submission. Use this evidence to discuss exactly when a warning arrives and how false alerts would affect a human inspection workflow.
