#!/bin/sh
# qb2 leg of the reblock_permute landing gate, card 1, run sequentially so one device context is
# open at a time. Every step writes its own log and its own JSON, so a pass that ends mid-campaign
# leaves finished steps intact. cwd is this worker's own worktree (fleet hygiene tears down a
# concluded slug's worktree, and a job rooted in someone else's loses its files mid-run).
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-permute-flip-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-flip-land
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa

L=perf/z_flip_land/logs
mkdir -p "$L"
T=examples/prot.yaml,perf/size512/fixtures/cdk2x2_298.yaml,perf/size512/fixtures/cdk2x2_384.yaml,perf/size512/fixtures/cdk2x2_512.yaml

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
  step "sweep_$M" $PY -u perf/z_flip_land/census_sweep.py --model "$M" --targets "$T" \
      --out "perf/z_flip_land/sweep_${M}_qb2c1.json"
done
step boltzgen $PY -u perf/y_permute_crossmodel/boltzgen_ab.py --aa-rounds 0 --rounds 0 \
    --out perf/z_flip_land/boltzgen_binder_qb2c1.json
step gate_crossmodel $PY -u scripts/release_gate.py --model openfold3 --model boltz2 \
    --model opendde --model boltzgen
step gate_capacity $PY -u scripts/release_gate.py --model capacity
step gate_abag $PY -u scripts/release_gate.py --model opendde-abag
echo "ALL_DONE $(date -u +%FT%TZ)"
