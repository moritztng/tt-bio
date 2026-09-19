#!/usr/bin/env bash
# Three gate arms in THE CONFIGURATION THAT ACTUALLY SHIPS: cond-hoist on, region T off.
#
# Arm 1 is the control that settles the l1-budget red. Arms 2 and 3 are cond-hoist's first
# cross-model gate coverage: c13-land-first's 14-arm green gate imported the shared checkout at
# 480ae2dfe, a tree with no _B2_DIT_COND_HOIST line in it at all, so the campaign's only shipped
# lever has direct Boltz-2 accuracy evidence and zero gate coverage anywhere else. The _B2_ prefix
# is a misnomer: tt_bio/rf3/token_dit.py:85 builds DiffusionTransformer(atom_level=False), so RF3's
# token DiT takes the cond-hoist branch too.
#
# With TT_BIO_TRIATT_B8=0 these are SINGLE-lever readings, not a stack, so a green isolates
# cond-hoist rather than saying "neither flag grossly breaks RF3".
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail-regiont
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT"
echo "=== SHIPPING-CONFIG GATE START $(date -Is) commit $(git rev-parse --short HEAD) card 3 ==="
for ARM in l1-budget rf3 rf3-1024aa; do
  echo
  echo "##### ARM $ARM START $(date -Is) #####"
  TT_BIO_TRIATT_B8=0 TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
    TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" scripts/release_gate.py --keep --model "$ARM" 2>&1 \
    | tee "perf/c14_bfp8/shipcfg_${ARM}.log"
  echo "##### ARM $ARM END rc=${PIPESTATUS[0]} $(date -Is) #####"
done
echo "=== SHIPPING-CONFIG GATE END $(date -Is) ==="
