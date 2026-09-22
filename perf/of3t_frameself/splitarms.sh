#!/usr/bin/env bash
# of3t-frameself: the s-path and z-path halves of the injected float64 arm.
#
# The producer is perf/of3t_trunkg043/ref_grad.py UNCHANGED -- no fork, no new argument. The only
# thing that differs from of3t-twoside's `arms.sh ctrl` is --cap-last, which points at a cotangent
# file with one of the two sides zeroed (perf/of3t_frameself/split_cot.py). The trunk gradient is
# linear in the cotangent, so the two arms sum to the ctrl arm exactly, and that sum is checked.
#
#   splitarms.sh sonly    cot_z = 0, so every gradient arrives through the SINGLE output
#   splitarms.sh zonly    cot_s = 0, so every gradient arrives through the PAIR output
set -uo pipefail
R=/home/ttuser/of3t_frameself
S=/tmp/of3t/of3t-frameself
T=/home/ttuser/of3t_refprec
PY=/home/ttuser/tt-bio-dev/env/bin/python
G=/home/ttuser/.coworker/wt/of3t-frameself/perf/of3t_trunkg043/ref_grad.py
mkdir -p "$R" "$S"
export PYTHONPATH="$T/of3pkg043:$T/deps:$T/pylibs"
cd "$R"

B=/home/ttuser/of3t_modelframe/boundary_model_n384.pt
echo "583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2  $B" | sha256sum -c - || exit 2

case "${1:?usage: splitarms.sh sonly|zonly}" in
  sonly) NM=split_sonly; C=$S/cot_sonly_n384.pt
         SHA=d5a372390b09c074b0e37b9ec6c15a86723079a76723595dc6299946de4c81bb ;;
  zonly) NM=split_zonly; C=$S/cot_zonly_n384.pt
         SHA=491f58645136d1bf7d82a13bd7d42cab7d7cdaa5554129dccf3e3ee89bb6552a ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
# A pipeline would run the case in a subshell and lose NM, so the digest check is its own line.
echo "$SHA  $C" | sha256sum -c - || exit 2

TH=${THREADS:-14}
echo "=== $NM start $(date -u +%FT%TZ) host $(hostname) threads $TH ==="
S0=$(date +%s)
OMP_NUM_THREADS=$TH nice -n 10 "$PY" "$G" \
  --tree "$T/of3pkg043" --boundary "$B" --cap-last "$C" \
  --policy f64 --blocks 48 --crop 384 --threads "$TH" --checkpoint \
  --out "$R/$NM.pt" --report "$R/$NM.json" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach'
rc=${PIPESTATUS[0]}
echo "=== $NM exit $rc elapsed $(($(date +%s)-S0))s $(date -u +%FT%TZ) ==="
exit "$rc"
