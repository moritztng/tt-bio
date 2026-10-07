#!/bin/bash
# The host arm after the key reset and dropout fix, run twice with different call ORDERS.
# If one_hot_weight=1.0 reads the same as the 5th call of a sweep and the 2nd call of a pair, the
# comparison no longer depends on how many rounds preceded it, which is what the fix claims.
set -u
cd /home/ttuser/.coworker/wt/bci-accept
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:/home/ttuser/.coworker/wt/bci-accept
export JAX_PLATFORMS=cpu
PY=/home/ttuser/fdv_fresh/venv/bin/python
COMMON="--states .bci/harden_states_60.pkl --af2-weights /home/ttuser/bcx_e2e/af2_params \
 --target-pdb /home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb --binder-length 60"
echo "=== sweep, 1.0 is the 5th call ==="
nice -n 19 timeout 5400 $PY perf/bci_accept/harden_forward_ab.py $COMMON --one-hot-weights 0.0 0.25 0.5 0.75 1.0
echo "=== pair, 1.0 is the 2nd call ==="
nice -n 19 timeout 5400 $PY perf/bci_accept/harden_forward_ab.py $COMMON --one-hot-weights 0.0 1.0
echo "=== 1.0 alone, the 1st call ==="
nice -n 19 timeout 5400 $PY perf/bci_accept/harden_forward_ab.py $COMMON --one-hot-weights 1.0
echo "=== done rc=$? ==="
