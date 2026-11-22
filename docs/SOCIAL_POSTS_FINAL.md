# Final public posts — MUSITU AXIOM Sentinel (TimeNet Round 1)

Publish after the updated demo is live. Every number below comes from [OPENTIME_5B_VERIFIED_GPU_RESULTS.md](OPENTIME_5B_VERIFIED_GPU_RESULTS.md). Attach 1–2 of the screenshots listed at the end. After posting, paste the post URLs into the Tally form and into [TIMENET_ROUND1_FINAL_SUBMISSION.md](TIMENET_ROUND1_FINAL_SUBMISSION.md).

## X (under 280 characters)

We fine-tuned the 5B OpenTime time-series LLM to warn when a NASA simulated jet engine is within 30 cycles of failure.

100 held-out engines: F1 0.94 avg over 3 seeds, 0 false alarms. Classical baseline: 0.89.

Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/
#TimeNet #OpenTime

## LinkedIn

Can a time-series language model warn a maintenance team before an engine fails, and beat a strong classical model at it?

For the TimeNet / OpenTime hackathon we built MUSITU AXIOM Sentinel on NASA's C-MAPSS turbofan data (simulated engines). We added the dataset to TimeNet as a connector, then fine-tuned the organizer's 5B OpenTime model to answer one question: will this engine reach end of life within the next 30 cycles?

Our first attempt trained on 16 examples and scored F1 0.32. It saw only the last 30 cycles, and the model rescales every series, which hid the slow wear trend. So we gave it each engine's full history, the 14 sensors that actually move, and the engine's age, and trained on all 80 development engines.

Results on NASA's 100 official test engines (25 at risk), each seed scored once:
• OpenTime 5B: F1 0.958 / 0.958 / 0.913 across three training seeds, zero false alarms
• Classical gradient boosting: F1 0.889 (5 at-risk engines missed)
• Our specialised temporal Transformer: F1 0.960

What we don't claim: 100 engines is small, and the bootstrap interval on the gain still touches zero. The data is simulated, so this says nothing about real aircraft. Thresholds and epochs were chosen on separate validation engines, never on the test set.

Everything is public: GPU logs, per-seed metrics, code and the TimeNet PR.

🌐 Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/
🧪 Code and evidence: https://github.com/evansmusitu/musitu-timenet-sentinel
🔀 TimeNet PR: https://github.com/OpenTSLM/TimeNet/pull/836

#TimeNet #OpenTime #TimeSeries #PredictiveMaintenance #MachineLearning

## Discord (showcase channel)

**MUSITU AXIOM Sentinel**: 30-cycle failure warnings on NASA C-MAPSS FD001 (simulated turbofans) with a fine-tuned `opentime/tslm-nov26`.

- What worked: full past-only engine history instead of a 30-cycle window (the per-series rescaling hid wear), 14 varying sensors, cycle count in the prompt, all 80 dev engines, 3 epochs. Threshold on log P(Yes) − log P(No), picked on validation engines.
- 100 official test engines: F1 0.958 / 0.958 / 0.913 over 3 seeds, 0 false alarms, 0 invalid answers. Gradient boosting baseline: 0.889. First 16-example pilot: 0.324.
- About 25 minutes per seed on one RTX PRO 6000 (Nebius).
- Caveats: n=100, bootstrap CI on the gain touches zero; simulated data.

Demo: https://evansmusitu.github.io/musitu-timenet-sentinel/ · Repo: https://github.com/evansmusitu/musitu-timenet-sentinel · TimeNet PR: https://github.com/OpenTSLM/TimeNet/pull/836

Feedback welcome, especially on harder subsets like FD004.

## Screenshots to take (organizers ask for metric screenshots)

1. **Demo, OpenTime section:** https://evansmusitu.github.io/musitu-timenet-sentinel/#opentime — the "100 official test engines · 25 at risk" panel with the three seeds and the baseline. Use this one on X and LinkedIn.
2. **Demo hero and stat strip:** top of https://evansmusitu.github.io/musitu-timenet-sentinel/ showing "OpenTime 5B warning F1 0.943".
3. **Results table on GitHub:** the table near the top of https://github.com/evansmusitu/musitu-timenet-sentinel (README).
4. **Proof of the GPU run:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45021536671 showing both seed jobs green.
5. **Evidence file:** https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/evidence/opentime-v2/summary.json (scroll to `f1_threshold` and `caveats`).
6. **TimeNet PR:** https://github.com/OpenTSLM/TimeNet/pull/836 header.

Take them after the PR into `main` has merged and GitHub Pages has redeployed (refresh the demo and check that it says 0.943).
