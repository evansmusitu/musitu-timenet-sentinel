# OpenTime 5B v2: evaluation and training protocol (not yet run on GPU)

**Objective:** Improve the genuine organizer-style `opentime/tslm-nov26` model's warning recall and F1, not confuse its scores with the independent compact temporal model.

## Confirmed original limitation

The November 22 GPU pilot used **16 training examples, 8 validation examples, and 1 epoch** and produced **0.3243** warning F1 on 100 official NASA FD001 test engines (6 true positives, 6 false positives, 19 missed positives and 69 true negatives). See the [original 5B experiment](OPENTIME_5B_VERIFIED_GPU_RESULTS.md).

The existing [16-example GPU entrypoint](../scripts/opentime_nebius_job.sh) remains unchanged for provenance. It never invoked the improved 256/64 v2 sampling protocol.

## v2 design

- Immutable OpenTime checkpoint: `opentime/tslm-nov26`, revision `5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983`.
- Immutable pinned OpenTSLM helper source: revision `b83a9901d401e9e872ee9048fe61e6c9df9a5b78`, now enforced SHA256 `c19db31d2ce6e84a4ce02c9a0fdfe0869c53775f77b04dd4be5c893afd8335e2`.
- **256** balanced training windows (128 positive, 128 negative) from **80** development engines.
- **64** validation windows retaining natural 11 positive / 53 negative class prevalence from **20 other development engines**.
- **100 held-out official NASA test engines**, never included in the training/validation engine groups.
- One epoch using the pinned OpenTSLM helper's actual native trainable modules and original validation-loss best-checkpoint policy. **No new training results have been obtained yet.**

## New evaluation: answer-token likelihood

The normal Yes/No generation metric is retained. When `--option-likelihood` is enabled, the **same trained OpenTSLM model** also scores the negative log-likelihood (NLL) of exactly `Yes` and `No` under the same time-series prompt, without another surrogate classifier.

Define `margin = NLL(No) - NLL(Yes)`; positive favors `Yes`. Report two held-out experiments side by side:

1. Uncalibrated binary decision `margin >= 0`.
2. Calibrated binary decision `margin >= threshold`, where the threshold is selected *only* on the 64 development validation engines to maximize their F1. The threshold is then frozen before it is applied to the 100 official test engines.

The likelihood method is an **exploratory new evaluation protocol** and is not directly interchangeable with the previously reported generation F1. The previously used official test set has already been explored during project development, so this is **not** a newly untouched independent confirmation. Validation-only threshold selection and separate reporting reduce but cannot eliminate post-hoc optimism.

The implementation records all per-engine test scores and exact predictions so wrong and invalid decisions can be audited. A likelihood margin is **not** a calibrated probability of future equipment failure and not a causal explanation.

## Reproducibility and current gate

- [Pure decision and threshold module](../sentinel/opentime_decision.py)
- [Tests](../tests/test_opentime_decision.py)
- [Trainer with opt-in likelihood evaluation](../sentinel/opentime_train.py)
- [Pinned v2 container entrypoint](../scripts/opentime_nebius_v2_job.sh)
- [CPU-only protocol acceptance](../.github/workflows/opentime-v2-decision.yml)

The GPU container is **disabled by default** and cannot itself create cloud resources. Its external Nebius job must enforce a separate maximum runtime, prohibit charges outside explicitly authorized promotional credits, and persist trained adapter weights before the job disappears. The script prints model metrics, per-engine score rows and weight hashes to the job logs; **weight hashes alone are not durable trained parameters**. Do not claim the training checkpoint is saved merely because a log was recorded.

The new code has passed CPU tests and real-data cohort verification. There is **no new 5B F1** until a genuinely paid or voucher-funded GPU job finishes with verifiable retained evidence.
