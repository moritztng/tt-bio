#!/bin/sh
# Phase 3: the boltzgen bound, chained behind phase 2 on the same card.
#
# `examples/binder.yaml` designs an 80..120 binder against a 117-residue target, so its pair track
# is [1,256,256,*] -- the one shape the kernel loses on and the one the window already excludes.
# Zero calls served there is a statement about that target's size, not about boltzgen. This runs one
# design against 1ahw chain A (214 residues), which puts the pair track at 320 or 352, inside the L1
# window, and records the served count. That turns the exclusion into a bound.
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-permute-flip-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

while pgrep -f "sh perf/z_flip_land/run_qb2_phase2.sh" > /dev/null 2>&1; do sleep 60; done

export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-flip-land
export PYTHONPATH=$WT
L=perf/z_flip_land/logs
mkdir -p "$L"
[ -f "$L/boltzgen_1ahw.done" ] && exit 0
$PY -u perf/y_permute_crossmodel/boltzgen_ab.py --aa-rounds 0 --rounds 0 \
    --spec perf/z_flip_land/binder_1ahw_A.yaml \
    --out perf/z_flip_land/boltzgen_1ahw_qb2c1.json > "$L/boltzgen_1ahw.log" 2>&1 \
  && touch "$L/boltzgen_1ahw.done"
echo "PHASE3_DONE $(date -u +%FT%TZ)"
