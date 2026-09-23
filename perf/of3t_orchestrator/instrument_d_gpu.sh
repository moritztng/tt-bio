#!/usr/bin/env bash
# PROTOCOL section 2, instrument D: upstream's own end-to-end training test, on a CUDA box.
#
# It has never executed anywhere in this campaign. On every host we own it reports
# `2 skipped, "Requires cuda; found cpu"`, so the scoreboard records it as "not passed" with the
# skip reported rather than worked around. Ask 9365 approved one rented GPU hour to change that.
#
# It skips on TWO conditions, not one, and the second is easy to miss: no CUDA, and no local PDB
# subset (`datasets/train_pdb_subset.yaml` + `datasets/pdb_training_set`, gitignored, not fetched
# by CI). A GPU box alone therefore turns "skipped for CUDA" into "skipped for data" -- which
# would look like progress and be none. Both are handled here.
#
# Provisioning reuses of3t-reference's proven recipe (`setup_gpu_box.sh`): the repo is
# aqlaboratory/openfold-3, tag v0.5.0, commit c4771653c5d0a3ebb0b3af71b05efd64bc44ee86, asserted
# rather than assumed.
#
# Runs detached and banks everything under $ROOT (NOT /tmp -- K61: /tmp is scoped, not durable,
# and a sweep already cost another row a 26-minute capture).
set -x
ROOT=${ROOT:-/root/of3t_orch}
TAG=v0.5.0
COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
mkdir -p "$ROOT"; cd "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1
echo "=== START $(date -u +%FT%TZ) ==="

echo "=== LINK CHECK FIRST. A rented box can install fine and still be unable to download. ==="
# Pass 51 lost ~$2 and 28 minutes to a box whose `git clone` and `pip install` worked and which
# then pulled from S3 at 43 kB/s, from Cloudflare at 17 B/s and from PyPI at 0.00 MB/s. Every
# step after this one depends on the link, so it is measured BEFORE anything is provisioned,
# against a source that has nothing to do with the job. Under 5 MB/s here means stop: the
# 1.68 GB dataset alone would cost more than the box is worth.
# The first version of this check used speed.cloudflare.com/__down, which returns **HTTP 403
# and a 1-byte body** from a datacentre IP -- so it reported 17 B/s on a box whose real
# throughput was 12-14 MB/s, and I condemned a host on it. A link probe must therefore fetch a
# REAL file, follow redirects, and report the byte count, so a broken URL cannot masquerade as
# a dead link. It probes the source this job actually depends on.
_out=$(curl -sL --max-time 15 -o /dev/null \
       -w "%{size_download} %{speed_download} %{http_code}" \
       "https://openfold3-data.s3.amazonaws.com/pdb_training_set/dataset_caches/validation_cache_with_templates.json" 2>/dev/null)
set -- $_out; _sz=${1:-0}; _lk=${2:-0}; _code=${3:-000}; _lk=${_lk%%.*}
echo "LINK: ${_lk} B/s, ${_sz} bytes, HTTP ${_code} from the S3 bucket this job reads"
if [ "${_code}" != "200" ] || [ "${_sz:-0}" -lt 1000000 ]; then
  echo "FATAL: the link probe itself did not work (http ${_code}, ${_sz} bytes). Fix the probe"
  echo "       before judging the box -- a broken probe reads exactly like a dead link."
  exit 4
fi
# 2 MB/s, not 5: at 2 MB/s the 1.68 GB cache is 14 minutes, which is affordable. The floor is
# set by what the job can pay for, not by what a good link looks like -- and a single stream to
# S3 is slower than an aggregate anyway, so this is a conservative reading of the box.
if [ "${_lk:-0}" -lt 2000000 ]; then
  echo "FATAL: link is ${_lk:-0} B/s, under the 2 MB/s floor. Destroy this box and take another;"
  echo "       the 1.68 GB cache alone would take $(( 1680000000 / (${_lk:-1} + 1) / 60 )) minutes."
  exit 3
fi

echo "=== box stamp ==="
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
lscpu | head -12; free -g; df -h "$ROOT" | tail -1

echo "=== deps ==="
apt-get update -qq && apt-get install -y -qq git curl >/dev/null
pip install -q --no-input "pytorch-lightning>=2.1" ml-collections biotite "rdkit<2026" \
  pdbeccdutils kalign-python ijson dm-tree einops torchmetrics deepspeed boto3 pytest \
  2>&1 | tail -3

echo "=== upstream $TAG ==="
[ -d openfold-3 ] || git clone -q --branch "$TAG" https://github.com/aqlaboratory/openfold-3.git
cd openfold-3
git log -1 --format='commit %H %ci %s'
test "$(git rev-parse HEAD)" = "$COMMIT" || { echo "FATAL: not the pinned commit"; exit 2; }
pip install -q -e . 2>&1 | tail -5
python -c "import torch,openfold3;print('torch',torch.__version__,'cuda',torch.version.cuda,'dev',torch.cuda.get_device_name(0),'of3',openfold3.__version__ if hasattr(openfold3,'__version__') else '?')"
command -v run_openfold || echo "WARN: run_openfold console script absent -- the test shells out to it"

echo "=== the local PDB subset, which is the OTHER skip condition ==="
python scripts/datasets/generate_subset_cache.py 2>&1 | tail -20
python scripts/datasets/download_subset.py 2>&1 | tail -20
ls -la datasets/ 2>/dev/null | head
du -sh datasets 2>/dev/null

echo "=== clock sampler, DURING the run (PROTOCOL 4a: a figure without its clock is not one) ==="
( while true; do nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$ROOT/clocks.csv"; sleep 5; done ) &
CLOCKPID=$!

echo "=== INSTRUMENT D: smoke ==="
/usr/bin/time -v pytest openfold3/tests/test_training_full.py -k smoke -rs 2>&1 | tail -60
echo "SMOKE_EXIT=$?"

echo "=== INSTRUMENT D: full_subset ==="
/usr/bin/time -v pytest openfold3/tests/test_training_full.py -k full_subset -rs 2>&1 | tail -60
echo "FULL_EXIT=$?"

kill $CLOCKPID 2>/dev/null
echo "=== clocks during ==="
python - <<'PY'
import csv, statistics as st
rows=[r for r in csv.reader(open('/root/of3t_orch/clocks.csv')) if r]
sm=[float(r[0]) for r in rows]; util=[float(r[1]) for r in rows]
print(f"CLOCK: sm min {min(sm):.0f} / median {st.median(sm):.0f} / max {max(sm):.0f} MHz over {len(sm)} samples, polled DURING")
print(f"UTIL : median {st.median(util):.0f} %")
PY
echo "=== DONE $(date -u +%FT%TZ) ==="
