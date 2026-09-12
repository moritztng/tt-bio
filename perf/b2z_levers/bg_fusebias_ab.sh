#!/usr/bin/env bash
# Does TT_BIO_FUSE_BIAS_STACKS change a BoltzGen design?
#
# The flag was left default-off for one stated reason: BoltzGen has the same LayerNorm+Linear
# bias stacks in its diffusion conditioning, and nobody had run a design under it. This is that
# run. Three arms, one process each, all on the same card: two base arms so the A/A floor is
# measured rather than assumed, then the fused arm against them.
#
# BoltzGen has no --seed, so `seedsite/sitecustomize.py` seeds torch at interpreter start in
# this process and in the shard the CLI spawns. Without it the A/A leg is not zero and the A/B
# reads the sampler's RNG instead of the lever.
#
#   bash perf/b2z_levers/bg_fusebias_ab.sh <card>
set -u
card="${1:-0}"
WT=/home/ttuser/.coworker/wt/b2z-levers-default-on
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=$WT/perf/b2z_levers
W=$D/work
mkdir -p "$W"

run() {  # name  flag_value
  local name="$1" flag="$2" out="$W/out_$1"
  rm -rf "$out"
  echo "=== arm $name  TT_BIO_FUSE_BIAS_STACKS=$flag  $(date -u +%H:%M:%SZ) ==="
  TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_CARDS="$card" \
  TT_BIO_LEASE_HOLDER=worker:b2z-levers-default-on \
  PYTHONPATH="$D/seedsite:$WT" BG_AB_SEED=1234 TT_BIO_FUSE_BIAS_STACKS="$flag" \
  "$PY" -m tt_bio.main design "$W/bg256.yaml" --model boltzgen --out_dir "$out" \
        --num_designs 1 --steps design --devices "$card" --debug \
        > "$W/log_$name.txt" 2>&1
  echo "rc=$?"
  find "$out" -name '*.cif' -o -name '*.pdb' -o -name '*.fasta' | head -10
}

run base_0 0
run base_1 0
run fuse_0 1
echo "ALL ARMS DONE $(date -u +%H:%M:%SZ)"
