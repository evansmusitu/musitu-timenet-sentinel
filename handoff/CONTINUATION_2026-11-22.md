# MUSITU AXIOM Sentinel — Exact Continuation Handoff
**Prepared:** 2026-11-22 (Africa/Harare)
**Canonical source:** https://github.com/evansmusitu/musitu-timenet-sentinel
**Project:** TimeNet / OpenTime November 2026 hackathon
**Source snapshot:** The accompanying ZIP's `SOURCE/` directory is the complete tracked GitHub repository at the `SNAPSHOT_COMMIT.txt` SHA. Treat this as the source of truth, not old ChatGPT memory.
**Evidence protection:** Do not include GitHub Actions secret values, Nebius RSA private keys, personal tokens, downloaded NASA raw source archives, or licensed model weights in public outputs or this package.

## FIRST PRIORITY — Hackathon submission deadline
Official [TimeNet challenge](https://challenge.timenet.ai/) / [Round 1 Tally form](https://tally.so/r/J1qbyK).
**Deadline:** Sunday **2026-11-22 at 11:59 PM PST** = Monday **2026-11-23 at 09:59 Africa/Harare**.

The repository's current [submission pack](../SOURCE/docs/TIMENET_ROUND1_FINAL_SUBMISSION.md) records the official Tally receipt and project-specific X/LinkedIn/Discord links as **not yet confirmed**. **Do not assume submission has occurred.** Before submitting, inspect connected Gmail, relevant browser history/official receipt and the form to avoid duplicates. If no receipt exists, complete organizer-required posts and submit exactly one legitimate entry before the deadline. If an existing entry is found, update it rather than duplicating.

Required organizer deliverables:
1. Project name: **MUSITU AXIOM Sentinel — Predictive Maintenance with Time-Series Intelligence**.
2. Team: MUSITU AXIOM. Main contact: Evans Musitu. Team member: Tally Vans. Use only previously verified contact information; ask only if a field is genuinely missing.
3. GitHub: https://github.com/evansmusitu/musitu-timenet-sentinel
4. Working demonstration: https://evansmusitu.github.io/musitu-timenet-sentinel/
5. TimeNet upstream connector pull request: https://github.com/OpenTSLM/TimeNet/pull/836 (last verified **OPEN, ready for review, NOT MERGED**).
6. Correct evidence for real 5B GPU run: https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462 and log recovery https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004
7. Project-specific X, LinkedIn and Discord URLs: obtain actual published URLs, not draft links.

## What actually works
### NASA C-MAPSS FD001 dataset
- Simulated turbofan engine trajectories, **not** live aircraft safety evidence.
- 21 sensor streams + 3 operating settings; 30-cycle past-only windows.
- Training/validation split by engine identity, official held-out NASA test engines separate.
- NASA data downloaded on demand and checksum verified; do not redistribute raw archive.
- TimeNet connector PR #836 added engine-level TRAIN/VALIDATION and official TEST conversion. Latest verified [compatibility CI](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984278576) **SUCCEEDED**; earlier runs failed during debugging. PR still requires upstream review and license provenance clarification.

### Reproduced model performance — 100 official held-out engine endpoints
| System | RUL MAE (cycles) | Warning F1 | Notes |
|---|---:|---:|---|
| Classical gradient-boosting baseline | 19.4025 | 0.8889 | CPU baseline |
| Separate compact temporal QA Transformer | 19.0436 | 0.9600 | Original fixed-question architecture, **not** OpenTime |
| Actual organizer OpenTime TSLM November 2026 5B, one-epoch pilot | Not measured | 0.3243, before 0 | Binary warning QA only, trained with 16 windows, validated 8 |

The real 5B pilot's before/after held-out warning accuracy was **24% → 75%**; invalid responses **64 → 0**; recall after adaptation **0.24**. This is real but weaker than the separate CPU baselines. Never claim 5B beats the baselines or is suitable for real aircraft maintenance.

### Real Nebius GPU experiment completed
- Nebius Serverless AI Job: `aijob-e07ag71cvmbtjv2s7w` in **eu-south1**.
- RTX PRO 6000 Blackwell, **96 GB VRAM**. One-epoch job started **2026-11-22 04:07:55 UTC**, completed **04:13:37 UTC**; script runtime **321.94 seconds**.
- Model: `opentime/tslm-nov26`, immutable revision `5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983`; Gemma license.
- Nebius credentials stored in **GitHub Actions encrypted secrets**, names `NEBIUS_PRIVATE_KEY`, `NEBIUS_PUBLIC_KEY_ID`, `NEBIUS_SERVICE_ACCOUNT_ID`, `NEBIUS_PROJECT_ID`. Do not retrieve or publish values.
- [Successful GPU GitHub run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462); [original logs recovered and verified](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004).
- Evidence included in SOURCE/docs/evidence/opentime-nov26-fd001-gpu-pilot-metrics.json, source scripts in SOURCE/sentinel/opentime_train.py and SOURCE/scripts/opentime_nebius_job.sh.
- Separate GPU watchdog and capacity workflows exist. **Do not trigger new paid Nebius GPU runs or create duplicate job** without checking balance, current quota, strict stop/cleanup, and explicit new authorization. The latest verified single-run GPU pilot has completed.

## Current product structure
- `SOURCE/README.md` — full project description, reproducibility.
- `SOURCE/sentinel/` — telemetry parsing, feature extraction, classical benchmark, compact temporal QA and real 5B GPU runner.
- `SOURCE/tests/` — regression/unit tests for data, model adapter and GPU cost/watchdog.
- `SOURCE/docs/index.html` — published interactive GitHub Pages demo.
- `SOURCE/docs/TIMENET_ROUND1_FINAL_SUBMISSION.md` — full organizer-ready pitch, comparison table, copy-ready public posts and missing proof.
- `SOURCE/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md` — real 5B metrics, licensing caveats and links.
- `SOURCE/docs/evidence/` — verified machine-readable GPU metrics.
- `SOURCE/.github/workflows/` — CI baseline, training, Nebius authentication, GPU recovery, watchdog and TimeNet PR compatibility.
- `TIMENET_CONNECTOR_PR/` — three connector source files copied from the associated fork at a pinned PR head SHA; note they are **not** in the primary repository tree.
- `MANIFEST.json` and `CHECKSUMS_SHA256.txt` — provenance and integrity evidence for this backup.

## Current GitHub state / important references
- Canonical repo: https://github.com/evansmusitu/musitu-timenet-sentinel
- Official PR: https://github.com/OpenTSLM/TimeNet/pull/836 (open, not merged, latest read).
- [CPU baseline](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482)
- [Compact temporal QA](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819987425)
- [TimeNet PR connector compatibility success](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984278576)
- [OpenTime 5B job successful](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44983256462)
- [GPU logs / metrics recovery verified](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984604004)
- [Project demo](https://evansmusitu.github.io/musitu-timenet-sentinel/)

## Safety / integrity
- This is a **point-in-time** artifact; do not silently overwrite newer upstream changes in the GitHub repository.
- Check current `main` and PR SHA first when continuing; compare before modifying.
- Never copy, print, recover, attach, or upload key material from GitHub Actions secrets or Windows private files.
- The archive contains code, docs, public measurements and configuration, but **no raw NASA data, license-controlled 5B model weights, paid cloud assets or GitHub encrypted secret values**.
- Source code alone is not evidence of an official application receipt. Seek confirmation before saying submitted.
- Do not automatically run GPU workflows to reproduce the completed pilot without explicit spend authorization.

## Suggested next steps (in priority order)
1. **Verify official Tally submission and social-post receipts**. Check Gmail and official submission dashboard/form, then publish missing project-specific X/LinkedIn/Discord posts and submit exactly once.
2. Complete the upstream PR #836 review/merge process, preserving source-data license caveat.
3. Verify final GitHub Actions status and recent feedback.
4. Optionally improve the 5B model after the competition deadline with a separate approved budget; the current F1 0.324 is an honest small pilot.
5. Preserve this handoff ZIP in Google Drive and update the project README with final official links when verified.

## Exact command to paste into a NEW ChatGPT conversation

> Continue the MUSITU AXIOM Sentinel TimeNet/OpenTime November 2026 Hackathon project from the attached Google Drive backup ZIP `MUSITU_AXIOM_TimeNet_Sentinel_FULL_HANDOFF_2026-11-22.zip`. Read `HANDOFF/CONTINUATION.md`, `MANIFEST.json`, `SOURCE/README.md`, `SOURCE/docs/TIMENET_ROUND1_FINAL_SUBMISSION.md`, `SOURCE/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md`, and `SOURCE/docs/evidence/opentime-nov26-fd001-gpu-pilot-metrics.json` FIRST. Treat the pinned source revision in `SNAPSHOT_COMMIT.txt` as the baseline, check the live GitHub repo `evansmusitu/musitu-timenet-sentinel` before edits, and check PR `OpenTSLM/TimeNet#836`. Our real Nebius 96 GB RTX PRO 6000 OpenTime 5B GPU pilot finished with genuine warning F1 0.3243 after one epoch on 16 samples, verified against 100 official NASA held-out engines; do NOT rerun any billable GPU job without renewed explicit authorization. Highest priority: urgently verify and complete the official TimeNet Round 1 Tally submission plus required project-specific X, LinkedIn and Discord links before 2026-11-22 11:59 PM PST, without duplicate submissions. Continue autonomously and efficiently, verify all receipts, do not ask me to repeat facts contained in the archive, and give me exact results and outstanding blockers.
