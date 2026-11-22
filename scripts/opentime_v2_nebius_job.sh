#!/usr/bin/env bash
# Injected into a Nebius Serverless AI Job for the full OpenTime fine-tune. No credentials present.
set -Eeuo pipefail
set +x
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
echo "MUSITU_V2_JOB_START"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY'
import io,os,tarfile,urllib.request
from pathlib import Path
sha=os.environ["MUSITU_REVISION"]
assert len(sha)==40 and all(c in "0123456789abcdef" for c in sha)
url=f"https://github.com/evansmusitu/musitu-timenet-sentinel/archive/{sha}.tar.gz"
print("Downloading public pinned source commit",sha,flush=True)
with urllib.request.urlopen(url,timeout=80) as r:
    data=r.read(10_000_001)
if len(data)>=10_000_000:
    raise SystemExit("Unexpected source archive size")
dest=Path("/tmp/musitu-src"); dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tf:
    for m in tf:
        if m.issym() or m.islnk() or ".." in Path(m.name).parts or m.name.startswith("/"):
            raise SystemExit("Invalid source archive entry")
    tf.extractall(dest)
dirs=[x for x in dest.iterdir() if x.is_dir()]
assert len(dirs)==1
Path("/tmp/musitu-src-dir").write_text(str(dirs[0]))
PY
cd "$(cat /tmp/musitu-src-dir)"
python -m pip install -q -r requirements-opentime.txt
# Same immutable checkpoint revision as the verified pilot, for a like-for-like comparison.
REV=5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983
echo "MODEL_REVISION=$REV"
timeout --signal=TERM --kill-after=30s 75m python -m sentinel.opentime_v2_train \
    --revision "$REV" --seed "${MUSITU_SEED:-42}" --epochs 3 --train-budget-seconds 2400 --output artifacts/opentime_v2
echo "MUSITU_V2_JOB_SUCCESS"
