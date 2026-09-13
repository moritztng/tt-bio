#!/usr/bin/env bash
# One Boltz-2 fold with TT_BIO_ATOM_L1 in a chosen state, for the one question BoltzGen raised:
# the K = 294 clash is a property of the gate and the op, not of BoltzGen, so it should reach any
# model on this code path at the size where its own atom axis reads K = 294. Boltz-2's published
# ladder stops at 1024 aa, which is K = 266 -- two buckets under the cliff and never probed.
#
# `tt_bio/size_limits.py` publishes NO ceiling for boltz2 on wormhole_b0 and never refuses, and
# 1024/1088/1152/1300/1408/1536/1664 aa are all recorded as folding, so 1152 aa is a size a user
# gets today.
set -euo pipefail
WT="${WT:-$HOME/wt-bgatoml1}"
CARD="${CARD:-1}"
SIZE="${SIZE:-1152}"
ARM="${ARM:-l1}"
ROOT="${ROOT:-$HOME/scratch/b2fold_${SIZE}_${ARM}}"
mkdir -p "$ROOT"

# The fixtures ship their a3m beside the yaml under the fixture's own stem, but --msa_dir's cache
# is keyed by a hash of the sequence, so pointing at that directory finds nothing and the fold
# refuses rather than silently folding single-sequence. Name the a3m in the yaml instead, which is
# what keeps this at full MSA depth.
FIX="$WT/perf/size512/fixtures"
SPEC="$ROOT/cdk2x2_${SIZE}.yaml"
cp "$FIX/cdk2x2_${SIZE}.yaml" "$SPEC"
printf '      msa: %s\n' "$FIX/cdk2x2_${SIZE}.a3m" >> "$SPEC"

TT_BIO_ATOM_L1=$([ "$ARM" = l1 ] && echo 1 || echo 0) \
TT_BIO_ATOM_L1_TRACE=1 \
TT_BIO_LEASE_CARDS="$CARD" \
TT_BIO_LEASE_HOLDER=worker:b2z2-boltzgen-atoml1-leg \
TT_BIO_LEASE_DIR="${TT_BIO_LEASE_DIR:-$HOME/leases}" \
PYTHONPATH="$WT" \
env -u TT_METAL_DEVICE_PROFILER \
"$HOME/env/bin/python" -m tt_bio.main predict \
    "$SPEC" --out_dir "$ROOT/out" --device_ids "$CARD" --seed 0 2>&1 | tee "$ROOT/fold.log"
