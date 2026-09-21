#!/usr/bin/env bash
# of3t-hostleg: the diffusion arm, with and without the eight ref_atom_feature_embedder linears
# on the card. Same instrument as `perf/of3t_f64softmax/devgrad_f64.sh` runs; only the
# --device-refatom flag and the host differ.
#
#   gradarm.sh shipped   CONTROL. No flag. Must reproduce the published arm's 547 compared,
#                        median 1.250047e-01, worst 1.850397e+01, 459 over the bar. A control
#                        that does not reproduce means this host or this tree is not the one the
#                        published number was taken on, and nothing below it can be compared.
#   gradarm.sh refatom   THE ARM. 555 compared expected: 547 + the eight.
#   gradarm.sh break     --device-refatom --permute-cot. Structure k seeded with k+1's cotangent.
#                        The eight must MOVE, or the reading is not of the gradient.
#
# The captures are copied from qb2 (`/home/ttuser/of3t_hostleg/`); qb1 carries neither.
# --ref-tree '' because the upstream 0.4.3 package tree is not on this host: this arm's reference
# is `sub_boundary.pt`'s own `grad_f64`, already inside the capture and pinned by the capture's
# digest, so nothing here imports upstream. The model-denominator scoring against
# `grads_f64_043.pt` is `score_seventeen.py`, separately and by digest.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-hostleg
PY=/home/ttuser/tt-bio-dev/env/bin/python
CAP=/home/ttuser/of3t_hostleg/diffcap043
OUT=/home/ttuser/of3t_hostleg
case "${1:-}" in
  shipped) TAG=_hl_shipped; EXTRA=() ;;
  refatom) TAG=_hl_refatom; EXTRA=(--device-refatom) ;;
  break)   TAG=_hl_break;   EXTRA=(--device-refatom --permute-cot) ;;
  *) echo "usage: gradarm.sh {shipped|refatom|break}"; exit 2 ;;
esac
CLK="$OUT/aiclk${TAG}.tsv"
: > "$CLK"
NODE="/sys/class/tenstorrent/tenstorrent!${CARD}/tt_aiclk"
( while true; do printf '%s\t%s\n' "$(date +%s)" "$(cat "$NODE" 2>/dev/null)" >> "$CLK"; sleep 1; done ) &
SAMPLER=$!
S=$(date +%s)
echo "=== of3t-hostleg arm ${1}, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap "$CAP" --ref-tree '' \
    --out-dir perf/of3t_hostleg \
    --dump-per-tensor "${EXTRA[@]}" \
    --dump-grads "$OUT/device_grads${TAG}.pt"
rc=$?
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
echo "ARM_END ${1} rc=$rc elapsed $((E-S))s  $(date -u +%FT%TZ)"
awk -F'\t' -v s="$S" -v e="$E" '
  $1+0 >= s && $1+0 <= e && $2 != "" { n++; c=$2+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) print "AICLK: NO SAMPLES"; else
        printf "AICLK during the arm: n=%d mean=%.0f MHz min=%d max=%d\n", n, t/n, mn, mx }' "$CLK"
exit $rc
