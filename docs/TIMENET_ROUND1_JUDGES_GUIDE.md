# TimeNet Round 1 — 90-second judges' walkthrough

**MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence**  
Project: https://evansmusitu.github.io/musitu-timenet-sentinel/  
TimeNet connector: https://github.com/OpenTSLM/TimeNet/pull/836 (open for review; **not merged**)

## The 90-second demonstration

**0–15s — The problem.** "Imagine a maintenance analyst looking at a complex machine with 21 sensors and three operating settings. Can we identify an early warning without looking into its future? We use NASA C-MAPSS FD001 simulated engine histories to test that question. This is a research benchmark—not a deployed aviation system."

**15–35s — A decision, not just a metric.** Open the [100-engine decision audit](https://evansmusitu.github.io/musitu-timenet-sentinel/decision-audit.html). The independent classical baseline identifies 20 of 25 at-risk engines, missing five, with no false alarms. Our separate compact temporal warning head identifies 24 of 25 at-risk engines, missing one, with one false alarm. Then select **Engine 041** in the [interactive engine demo](https://evansmusitu.github.io/musitu-timenet-sentinel/#demo): true remaining life 18 cycles; classical prediction 38.68 (missed), compact prediction 28.42 (warning). This is one deliberately selected held-out example, not a typical case. The complete evaluation spans 100 engines.

**35–55s — TimeNet contribution.** Point to [upstream PR #836](https://github.com/OpenTSLM/TimeNet/pull/836). It adds a real C-MAPSS FD001 dataset connector with **30-cycle past-only windows**, **engine-disjoint development splits**, separate official held-out test engines, source checksums and no raw-data redistribution. The connector now also has 11 purpose-built synthetic tests, strict source/label validation, and a [successful actual TimeNet environment acceptance job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45002805232) including the real NASA dataset conversion. Upstream PR review and precise source license provenance remain unresolved.

**55–75s — The OpenTime 5B model, fine-tuned properly.** We fine-tuned the organizer's **5.24B-parameter OpenTime November 2026 checkpoint** on one **Nebius RTX PRO 6000 (96 GB)** using every FD001 development-train engine, the engine's **full past history**, the 14 sensors that vary, and the cycle count in the prompt. Epoch and warning threshold were chosen on 20 disjoint validation engines. On the **100 official test engines**, warning F1 is **0.958, 0.958 and 0.913** across three training seeds (mean 0.943), against **0.889** for the classical baseline, with **zero false alarms** in every seed. The model's own Yes/No answer scores 0.941–0.980 with no invalid outputs. Our earlier 16-example pilot scored 0.324; that is now superseded. Bootstrap intervals on the gain still touch zero, so we say "beats the baseline in every seed", not "proven superior". [Full GPU evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

**75–90s — What next.** "What makes Sentinel valuable is its auditable path from raw NASA telemetry to a TimeNet dataset, a fine-tuned time-series language model and a decision a maintenance analyst can inspect. A properly adapted OpenTime model now matches our best specialised model and beats the classical baseline on held-out engines. The next test is harder data with mixed operating conditions (FD004). We have not claimed deployment or safety certification."

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
| **OpenTime November 2026 5B v2 fine-tune** (3 seeds) | Not measured (binary QA only) | **0.958 / 0.958 / 0.913** (mean 0.943) | Actual organizer checkpoint, 80 training engines, 3 epochs |
| OpenTime 5B first pilot (superseded) | Not measured | 0.3243 | 16 training examples, 1 epoch |

## Exploratory simulated cross-domain follow-up — classical and compact models

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
- [OpenTime 5B v2 GPU run, seed 42](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45018761652) and [seeds 7 and 99](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021536671)
- [Machine-readable OpenTime v2 evidence](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/opentime-v2/summary.json)
- Superseded pilot: [Nebius job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462), [log recovery](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004)

### Boundaries of the claim

*NASA FD001 is simulated data, not live aircraft or real maintenance telemetry.* The demo displays selected held-out predictions from independent CPU-trained models; it does **not** execute the 5B checkpoint live. The OpenTime model predicts a binary 30-cycle warning (no RUL MAE). 100 test engines are too few for operational conclusions, and the seeds trained on 1,568, 1,190 and 1,418 examples because the script fits the training set to a time budget. No source NASA archive or licensed model weights are redistributed. The dataset license needs upstream clarification. PR #836 is still **open**, not merged.

**Competition submission status:** the team owner reported submitting the official Tally Round 1 form on November 22, 2026. We have not independently obtained an organizer project receipt or public social post permalinks, so do not imply external receipt verification. The entry should not be resubmitted.
