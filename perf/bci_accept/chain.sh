#!/bin/bash
# Wait for capture_logits.py to land its artifacts, then run the two card-free analyses that
# depend on them: the conditioning probe on real logits, and the host-JAX half of the harden A/B.
set -u
WT=/home/ttuser/.coworker/wt/bci-accept
PY=/home/ttuser/fdv_fresh/venv/bin/python
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:$WT
cd "$WT" || exit 1

for _ in $(seq 1 240); do
  [ -f .bci/harden_states.pkl ] && break
  sleep 15
done
if [ ! -f .bci/harden_states.pkl ]; then
  echo "harden_states.pkl never landed; capture did not reach harden"
  exit 1
fi

echo "=== conditioning probe on the real captured logits ==="
nice -n 10 timeout 1800 "$PY" perf/bci_accept/stage_conditioning.py --logits .bci/logits.npz

echo
echo "=== harden A/B, host-JAX arm alone (no card) ==="
nice -n 10 timeout 3600 "$PY" perf/bci_accept/harden_forward_ab.py \
  --states .bci/harden_states.pkl --af2-weights /home/ttuser/bcx_e2e/af2_params
echo "=== chain done rc=$? ==="
