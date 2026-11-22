# TimeNet official full-builder acceptance — 22 November 2026

## What was actually verified

A [successful GitHub Actions job](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45044257045) checked out the **exact TimeNet fork commit** `2eb9b5aea729d538e760e7601d863c8942bc4c6b` associated with [upstream connector PR #836](https://github.com/OpenTSLM/TimeNet/pull/836).

Using the pinned TimeNet workspace lockfile and the public NASA C-MAPSS FD001 source archive:

1. All **11** synthetic regression tests for the submitted connector passed (one unrelated pytest-deprecation warning).
2. The connector was discovered under **`musitu/cmapss-fd001`** and metadata preserved its `license: OTHER` uncertainty.
3. The **official** `timenet-build build` CLI created **release `musitu/cmapss-fd001@0.1.0`** from real source data.
4. The resulting local registry contained four release files with **`.duckdb`, `.parquet` and `.json`** extensions.
5. A second official CLI invocation recognized the existing 0.1.0 release and returned **"Nothing to do"**, demonstrating repeatable release resolution.

Example reproducible commands, from the exact fork commit and its TimeNet workspace (the command uses an isolated local registry, not a public data upload):

```bash
uv sync --locked --all-packages --no-dev
uv pip install --python .venv/bin/python 'pytest>=8,<10'
uv run --no-sync python -m pytest -q packages/timenet-connectors/src/timenet_connectors/datasets/musitu/cmapss_fd001/tests
uv run --no-sync timenet-build build musitu/cmapss-fd001 --no-isolation --local-registry /tmp/musitu-full-build-registry
uv run --no-sync timenet-build build musitu/cmapss-fd001 --no-isolation --local-registry /tmp/musitu-full-build-registry
```

The workflow source is [`.github/workflows/timenet-full-build.yml`](../.github/workflows/timenet-full-build.yml), and the pinned original upstream connector code is [PR #836](https://github.com/OpenTSLM/TimeNet/pull/836).

## What this does not prove

- **Not upstream acceptance:** PR #836 is open and requires OpenTSLM maintainer review; the full build succeeded in our GitHub Actions environment, not in the official upstream merge gate.
- **Not permission to redistribute:** the NASA/Zenodo record does not specify an unambiguous SPDX grant. The dataset metadata intentionally says `OTHER`; user-facing artifacts do not redistribute raw NASA telemetry. A local test registry was generated and discarded.
- **Not an OpenTime model score:** the TimeNet builder tests dataset ingestion and serialization, not the organizer's 5B classifier. The original genuine OpenTime pilot's held-out F1 remains 0.3243.
- **Not deployment validation:** NASA FD001 is simulated turbofan data, and no operational aircraft safety claim is made.
