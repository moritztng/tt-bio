#!/bin/sh
# Phase 2 of the qb2 landing gate, chained behind run_qb2.sh on the same card so only one device
# context is ever open. Waits for the phase-1 driver to exit rather than racing it.
#
# What this adds that phase 1 does not: the flag-OFF control at the sizes that actually SERVE calls.
# The flip's bit-exactness is settled at 24 window cells, but the brief asks for it re-taken rather
# than inherited, and comparing the written CIF sha between arms at 298 (L1 leg) and 384 (DRAM leg)
# is the strongest form of that -- full coordinates, not a scalar. It also covers boltz2, whose
# metrics dict carries no top-level plDDT, so the CIF sha is the only structural evidence there.
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-permute-flip-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

while pgrep -f "sh perf/z_flip_land/run_qb2.sh" > /dev/null 2>&1; do sleep 60; done

export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-flip-land
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa

L=perf/z_flip_land/logs
mkdir -p "$L"
T=perf/size512/fixtures/cdk2x2_298.yaml,perf/size512/fixtures/cdk2x2_384.yaml

step() {
  name=$1; shift
  [ -f "$L/$name.done" ] && { echo "SKIP $name"; return; }
  echo "=== START $name $(date -u +%FT%TZ)"
  "$@" > "$L/$name.log" 2>&1
  rc=$?
  echo "=== END $name rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}

for M in openfold3 boltz2 opendde; do
  step "control_$M" $PY -u perf/z_flip_land/census_sweep.py --model "$M" --targets "$T" --control \
      --out "perf/z_flip_land/control_${M}_qb2c1.json"
done

# DEFAULT mode, and no --seeds at all: a bare integer parses as the single-element list [5], no
# fixture carries that seed, every leg comes back BLOCKED-REF-REGEN-NEEDED and the script still
# prints GATE PASS. Read the per-leg verdicts, never the banner.
step fpg_hsa $PY -u scripts/full_parity_gate.py --workers "tt-quietbox2:1" \
    --leg protenix-hsa-msa --leg boltz2-hsa-nomsa
echo "PHASE2_DONE $(date -u +%FT%TZ)"
