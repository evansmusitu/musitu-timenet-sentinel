#!/usr/bin/env bash
# Injected into a Nebius Serverless AI Job. No credentials are present.
set -Eeuo pipefail
set +x
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
echo "MUSITU_OPENTIME_JOB_START"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY'
import io,os,subprocess,tarfile,urllib.request
sha=os.environ["MUSITU_REVISION"]
assert len(sha)==40 and all(c in "0123456789abcdef" for c in sha)
url=f"https://github.com/evansmusitu/musitu-timenet-sentinel/archive/{sha}.tar.gz"
print("Downloading public pinned source commit",sha,flush=True)
with urllib.request.urlopen(url,timeout=80) as r:
    data=r.read(10_000_001)
if len(data)>=10_000_000:
    raise SystemExit("Unexpected source archive size")
from pathlib import Path
dest=Path("/tmp/musitu-src")
dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tf:
    # Reject unusual archive entry paths and symbolic links
    for member in tf:
        if member.issym() or member.islnk() or ".." in Path(member.name).parts or member.name.startswith("/"):
            raise SystemExit("Invalid source archive entry")
    tf.extractall(dest)
dirs=[x for x in dest.iterdir() if x.is_dir()]
assert len(dirs)==1
Path("/tmp/musitu-src-dir").write_text(str(dirs[0]))
PY
cd "$(cat /tmp/musitu-src-dir)"
python -m pip install -r requirements-opentime.txt
python - <<'PY'
import torch
print("TORCH_VERSION",torch.__version__,"CUDA_VERSION",torch.version.cuda,flush=True)
assert torch.cuda.is_available(),"CUDA unavailable"
print("CUDA_DEVICE",torch.cuda.get_device_name(0),flush=True)
PY
REV="$(python - <<'PY'
from huggingface_hub import model_info
m=model_info("opentime/tslm-nov26")
print(m.sha)
PY
)"
echo "MODEL_REVISION=$REV"
# Start with a small but genuine adaptation; no synthetic 'OpenTime' claims.
timeout --signal=TERM --kill-after=20s 43m python -m sentinel.opentime_train \
    --revision "$REV" --epochs 1 --train-limit 16 --val-limit 8 \
    --output artifacts/opentime
if [[ -f artifacts/opentime/metrics.json ]]; then
  echo "MUSITU_OPENTIME_METRICS_BEGIN"
  cat artifacts/opentime/metrics.json
  echo
  echo "MUSITU_OPENTIME_METRICS_END"
fi
echo "MUSITU_OPENTIME_JOB_SUCCESS"
