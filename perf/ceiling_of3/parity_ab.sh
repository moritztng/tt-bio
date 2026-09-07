#!/bin/sh
# Bit-exact A/B of one OpenFold3 fold between two trees, with an A/A control.
#
#   sh parity_ab.sh <tag> <rung> <tree> [seed]
#
# MODEL, RUN, RUNGS, CARD and HOLDER override the defaults, so the same A/B scores any model on
# this trunk. With CARD set the chip is pinned and the picker never runs -- which is what a box
# with 26 chips serving users needs, because a picker that scans is a picker that can land on
# one of them.
#
# The PairWeightedAveraging change replaces an out-of-place `ttnn.add` with `ttnn.add_` on the
# same operands in the same order, so it should not move a single bit. "Should not" is an
# argument; this measures it. Bit-exactness is a stronger statement than a PCC against a
# reference, because if no bit moves then every reference comparison is unchanged by
# construction -- which is exactly the claim the bar asks for at 128/256/512.
#
# An A/A leg (the same tree run twice) is mandatory and runs first: without it a matching A/B
# proves only that the fold is deterministic, and a differing A/B could just be run-to-run noise.
set -u
tag=$1; rung=$2; tree=$3; seed=${4:-42}
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-openfold3}
RUN=${RUN:-/home/cust-team/mthuening/ceilof3/rundir}
RUNGS=${RUNGS:-msafix_tile}
HOLDER=${HOLDER:-worker:ceiling-openfold3}
OUT=$RUN/parity/$tag
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
cd "$RUN" || exit 1
mkdir -p "$OUT"
# The picker comes from the HARNESS, never from the tree under test: the A leg is plain
# `main`, which does not carry pick_chip.py at all and whose tt_bio.runtime has no
# card-to-node map, so asking the tree under test to find a chip would fail on the arm whose
# numbers matter most.
if [ -n "${CARD:-}" ]; then
  dev=$CARD
else
  dev=$("$PY" "$HERE/pick_chip.py") || { echo "$tag: no free chip"; exit 3; }
fi
# TT_BIO_SIZE_LIMIT=0: a parity rung is chosen for what it exercises, not for whether the
# published cap happens to admit it.
TT_BIO_SIZE_LIMIT=0 TT_VISIBLE_DEVICES="$dev" TT_BIO_LEASE_CARDS="$dev" TT_BIO_LEASE_HOLDER="$HOLDER" \
  TT_BIO_LEASE_TIMEOUT=20 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$tree" \
  "$PY" -m tt_bio.main predict "$RUNGS/$rung.yaml" --model "$MODEL" \
  --accelerator tenstorrent --out_dir "$OUT" --override --msa_dir msacache_deep --msa_cache_only \
  --seed "$seed" > "$OUT.log" 2>&1
echo "$tag rc=$? card=$dev tree=$(git -C "$tree" rev-parse --short HEAD) $(date -u +%FT%TZ)"
