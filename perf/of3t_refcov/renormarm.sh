#!/usr/bin/env bash
# of3t-refcov: the two arms the composition adds, in BOTH tree states the union scores.
#
# `renorm` in this campaign is the softmax-backward row-sum repair (of3t-apbgrad, D56). It is
# NOT a harness flag any more: `tt_bio/autograd.py:87` reads
# `env_flag("TT_BIO_SOFTMAX_BW_RENORM", True)` since 2de9355d1, so the repair is ON by default
# and `device_gradient.py --softmax-bw-renorm` is a no-op on this tree. The only way to get the
# union's `shipped` arm is to turn it OFF explicitly.
#
# Which is why both of3t-hostleg dumps are RENORM arms whatever their filenames say: they ran
# 2026-09-21T21:59Z, after D56, and both stamped `softmax_bw_renorm_live: true`. This script
# produces their `renorm=off` twins so the 17 enter both arms of the union instead of entering
# one and being scored as zero in the other.
#
#   renormarm.sh {ie|diffusion} {on|off}
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-refcov
ARM=${1:-}; RN=${2:-on}
case "$RN" in on) export TT_BIO_SOFTMAX_BW_RENORM=1 ;; off) export TT_BIO_SOFTMAX_BW_RENORM=0 ;;
   *) echo "usage: renormarm.sh {ie|diffusion} {on|off}"; exit 2 ;; esac
PY=/home/ttuser/tt-bio-dev/env/bin/python
SCRATCH=/tmp/of3t/of3t-refcov
mkdir -p "$SCRATCH"
CLK="$SCRATCH/aiclk_${ARM}_${RN}.tsv"
: > "$CLK"
NODE="/sys/class/tenstorrent/tenstorrent!${CARD}/tt_aiclk"
( while true; do printf '%s\t%s\n' "$(date +%s)" "$(cat "$NODE" 2>/dev/null)" >> "$CLK"; sleep 1; done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT
S=$(date +%s)
echo "=== of3t-refcov arm=$ARM renorm=$RN card=$CARD  $(date -u +%FT%TZ) ==="
case "$ARM" in
  ie)
    "$PY" perf/of3t_hostleg/ie_arm.py \
        --boundary /home/ttuser/of3t_hostleg/ie_boundary.pt \
        --out "perf/of3t_refcov/IE_ARM_f64_renorm_${RN}.json" \
        --dump "$SCRATCH/ie_grads_f64_renorm_${RN}.pt"
    ;;
  diffusion)
    "$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "_rc_refatom_${RN}" \
        --cap /home/ttuser/of3t_hostleg/diffcap043 --ref-tree '' \
        --out-dir perf/of3t_refcov --dump-per-tensor --device-refatom \
        --dump-grads "$SCRATCH/device_grads_rc_refatom_${RN}.pt"
    ;;
  *) echo "usage: renormarm.sh {ie|diffusion} {on|off}"; exit 2 ;;
esac
rc=$?
E=$(date +%s)
echo "ARM_END $ARM renorm=$RN rc=$rc elapsed $((E-S))s  $(date -u +%FT%TZ)"
awk -F'\t' -v s="$S" -v e="$E" '
  $1+0 >= s && $1+0 <= e && $2 != "" { n++; c=$2+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) print "AICLK: NO SAMPLES"; else
        printf "AICLK during the arm: n=%d mean=%.0f MHz min=%d max=%d\n", n, t/n, mn, mx }' "$CLK"
exit $rc
