# MUSITU AXIOM Sentinel — TimeNet / OpenTime Hackathon

Evidence-first predictive maintenance prototype using NASA Ames C-MAPSS FD001 simulated turbofan engine telemetry.

**Status:** CPU benchmark and tests implemented. OpenTime 5B checkpoint is **not trained or verified** here. The current competition aims to integrate a real TimeNet/TimeF dataset and a trained or fine-tuned time-series language model.

## Live evidence

- **Interactive demonstration:** https://evansmusitu.github.io/musitu-timenet-sentinel/
- **Successful official NASA FD001 CPU benchmark:** https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44819062482
- **Download artifact:** open that run's Artifacts section and download `nasa-cmapss-results`.
- **Official test on 100 unseen engines:** 19.4025-cycle RUL MAE; 27.2319-cycle RMSE; 30-cycle warning F1 0.8889, precision 1.0, recall 0.8.
- **Separate experimental fixed-prompt temporal QA Transformer:** source `sentinel/temporal_qa.py`, workflow `.github/workflows/temporal.yml`. Any metrics must come from a completed, successful model-training run. This is **not** OpenTime's 5B model or a general-purpose TSLM.

## Reproduce

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python -m sentinel.benchmark --download
```

The runner fetches the NASA FD001 archive from the versioned Zenodo record and verifies its MD5, then produces `artifacts/metrics.json`, `artifacts/official_test_predictions.csv` and `artifacts/summary.md`.

**Method:** 30-cycle past-only windows, 21 sensor streams, 3 operating settings, engine-group-disjoint development split, held-out NASA test engines, gradient-boosted regression of remaining useful life (RUL) plus 30-cycle warning classification. Baseline is classical machine learning, not the OpenTime TSLM.

The experimental checkpoint shared in OpenTime Discord is https://huggingface.co/opentime/tslm-nov26 . Martin noted that novel model components are not yet included in the main OpenTSLM repository; inspect the example helpers at https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/examples/opentslm-finetune-arfbench/scripts/utils.py .

## Source

NASA Ames Prognostics Center of Excellence: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Archived source: https://zenodo.org/records/15346912 (MD5 `79a22f36e80606c69d0e9e4da5bb2bdf`). We do **not** redistribute source data.

TimeNet: https://github.com/OpenTSLM/TimeNet . Current challenge: https://challenge.timenet.ai/ . Submission deadline 2026-11-22 11:59 p.m. PST, official form https://tally.so/r/J1qbyK .

**Limitations:** Engine data is simulated. Do not use these predictions for real aircraft maintenance. Results from a CPU baseline must not be presented as TSLM results. An upstream TimeNet PR, real TSLM training/evaluation, hosted demo, social posts and submission remain to be completed.
