#!/bin/bash
# Sequential accuracy-arm driver for shared-softmax-crossmodel. One arm at a time on one card,
# so the device lease is never contended by this task with itself. Rooted in this worker's own
# worktree on purpose: a job rooted in another slug's worktree gets its files deleted mid-run.
cd /home/ttuser/.coworker/wt/shared-softmax-crossmodel || exit 1
export PYTHONPATH=/home/ttuser/.coworker/wt/shared-softmax-crossmodel
export TT_BIO_LEASE_HOLDER=worker:shared-softmax-crossmodel
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-2}

run_arm () {   # run_arm <name> <ab-tokens-or-empty>
  local name=$1 ab=$2 try=0
  while [ $try -lt 40 ]; do
    try=$((try+1))
    echo "[$(date -u +%H:%M:%S)] $name attempt $try"
    if TT_BIO_ACCURATE_SOFTMAX_AB="$ab" "$PY" scripts/full_parity_gate.py \
         --leg openfold3-ubq-msa --workers tt-quietbox2:$CARD \
         --out perf/xmsoftmax/results/anchor_of3_${name}.json \
         --workdir /tmp/xmsm/of3_${name} >> /tmp/xmsm/of3_${name}.log 2>&1; then
      echo "[$(date -u +%H:%M:%S)] $name OK"; return 0
    fi
    if grep -q DeviceInUseError /tmp/xmsm/of3_${name}.log; then
      echo "[$(date -u +%H:%M:%S)] $name: card busy, retrying in 120s"; sleep 120; continue
    fi
    echo "[$(date -u +%H:%M:%S)] $name FAILED (not a lease collision)"; return 1
  done
  return 1
}

run_arm on "openfold3.trunk,openfold3.confidence,openfold3.template,openfold3.msa"
echo "[$(date -u +%H:%M:%S)] driver done"
