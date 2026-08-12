#!/bin/bash
# ONE benchlock hold for the whole §8 exec order: five sibling perf worktrees fold on this box, so
# re-acquiring the lock per step is the thing most likely to lose the measurement.
WT=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf PYTHONPATH=$WT
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export BENCHLOCK_WAIT_S=1500 BENCHLOCK_LOAD_WAIT_S=600

/home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- bash -c '
set -x
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd /home/ttuser/.coworker/wt/boltz2-512aa-deep-perf

echo "=== STEP 0: S0 split census (the gate) ==="
$PY -u perf/b2deep/decompose.py --model boltz2 --sizes 512 --deep --split \
    --arms on --out perf/b2deep/decomp_split_512.json; echo "S0 RC=$?"

echo "=== STEP 2: S1 fused-SDPA screen, on,sdpa,on ==="
$PY -u perf/other512/fold_ab_multi.py --model boltz2 --sizes 512 \
    --arms on,sdpa,on --out perf/b2deep/ab_s1_sdpa_512.json; echo "S1 RC=$?"

echo "=== STEP 1: roofs on card 1 ==="
$PY -u perf/other512/roofs.py perf/b2deep/roofs_card1.json; echo "ROOFS RC=$?"

echo "=== STEP 3: S2 atom-pad off-fold screen ==="
$PY -u perf/b2deep/s2_atom_pad.py perf/b2deep/s2_atom_pad.json; echo "S2 RC=$?"
' 2>&1
echo "WRAPPER RC=$?"
