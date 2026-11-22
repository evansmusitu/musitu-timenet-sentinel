#!/usr/bin/env bash
# Genuine OpenTime v2 GPU container entrypoint only. NEVER creates cloud
# resources. CI only syntax-checks it; separate provider-enforced credit-only
# authorization is required before renting any GPU.
set -Eeuo pipefail
set +x
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ "${MUSITU_ALLOW_OPENTIME_V2_TRAINING:-}" != "explicitly-approved-credit-only" ]]; then
  echo 'OpenTime v2 GPU job not approved by this container environment.'
  exit 71
fi
: "${MUSITU_REVISION:?Pinned MUSITU source commit required}"
[[ "$MUSITU_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid source SHA"; exit 71; }

echo "MUSITU_OPENTIME_V2_START"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import io,os,tarfile,urllib.request
from pathlib import Path
sha=os.environ["MUSITU_REVISION"]
url=f"https://github.com/evansmusitu/musitu-timenet-sentinel/archive/{sha}.tar.gz"
with urllib.request.urlopen(url,timeout=90) as r:
    data=r.read(12_000_001)
if len(data)>12_000_000:
    raise SystemExit("Source snapshot too large")
dest=Path("/tmp/musitu-v2-src")
dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tf:
    for item in tf.getmembers():
        p=Path(item.name)
        if item.issym() or item.islnk() or p.is_absolute() or ".." in p.parts:
            raise SystemExit("Unsafe GitHub source archive member")
    tf.extractall(dest)
dirs=[d for d in dest.iterdir() if d.is_dir()]
if len(dirs)!=1: raise SystemExit("Unexpected source archive layout")
Path("/tmp/musitu-v2-snapshot").write_text(str(dirs[0]))
print("Immutable training source",sha)
PY
cd "$(cat /tmp/musitu-v2-snapshot)"
python -m pip install -r requirements-opentime.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(),"CUDA unavailable"
assert torch.cuda.get_device_properties(0).total_memory>=45*1024**3,"At least 48GB advertised VRAM required"
print("OPEN_TIME_V2_GPU",torch.cuda.get_device_name(0),flush=True)
PY

# Pin the EXACT checkpoint used in the November 22 genuine 5B pilot.
MODEL_REVISION="5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983"
echo "MODEL_REVISION=$MODEL_REVISION"
# External Nebius job must have its own independently enforced cloud-side timeout.
# This is NOT connected to any automatic GitHub push launch workflow.
timeout --signal=TERM --kill-after=20s 105m \
  python -m sentinel.opentime_train \
    --revision "$MODEL_REVISION" \
    --epochs 1 --train-limit 256 --val-limit 64 \
    --validation-sampling prevalence_matched \
    --option-likelihood \
    --output artifacts/opentime-v2

python - <<'PY'
from pathlib import Path
import hashlib,json
base=Path("artifacts/opentime-v2")
required=["metrics.json","validation_likelihood_predictions.json",
          "official_test_likelihood_predictions.json","before_predictions.json",
          "after_predictions.json","trained_parameters.safetensors"]
for name in required:
    path=base/name
    if not path.is_file() or path.stat().st_size<=0:
        raise SystemExit("Missing training evidence: "+name)
    print("ARTIFACT_SHA256",name,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
metrics=json.loads((base/"metrics.json").read_text())
likelihood=metrics.get("likelihood_evaluation") or {}
if metrics.get("train_examples")!=256 or metrics.get("validation_examples")!=64:
    raise SystemExit("Train/validation data preparation differs from 256/64")
if likelihood.get("official_test_used_for_threshold_selection") is not False:
    raise SystemExit("Official-test threshold leakage guard failed")
if likelihood.get("test_validation_selected_threshold",{}).get("count")!=100:
    raise SystemExit("Official 100-engine score evaluation missing")
print("MUSITU_OPENTIME_V2_METRICS_BEGIN",flush=True)
print(json.dumps(metrics,sort_keys=True),flush=True)
print("MUSITU_OPENTIME_V2_METRICS_END",flush=True)
for name in ("validation_likelihood_predictions.json","official_test_likelihood_predictions.json"):
    print("MUSITU_OPENTIME_V2_ROWS_BEGIN",name,flush=True)
    print((base/name).read_text(),flush=True)
    print("MUSITU_OPENTIME_V2_ROWS_END",name,flush=True)
print("MUSITU_OPENTIME_V2_COMPLETED",flush=True)
PY
