#!/usr/bin/env bash
# Is block 47's factor magnitude-dependent? The cotangent-scaling arm, no new capture needed.
#
# The backward is LINEAR in the injected cotangent, so in exact arithmetic scaling it by K scales
# every gradient by K and leaves norm_ratio and cos exactly unchanged. --cot-scale K therefore
# injects K*cot on the device and divides the device gradients by K again at scoring time, while
# leaving the reference side completely alone -- so ref_norm, the A14 floor, reach and the mass
# shares are bit-identical to the K=1 arm and ANY movement in the factors is finite-precision
# magnitude dependence and nothing else.
#
# Block 47's reference gradient total is 1.0961e-01 against 6.29e-03..1.56e-02 for the other six
# ladder blocks -- 7x the next highest, the one variable that separates it at n=7. K = 1/8 puts
# it at 1.37e-02, inside that range. K = 1/64 puts it an order below it.
#
# Registered before the run:
#   factors move toward 1.00 -> magnitude-dependent, and the dtype carrying it is next
#   factors stay at 1.1832 / 0.8827 -> magnitude is dead too, and it is not the gradient size
#
# The K=1 arm is re-run FIRST as a regression control: it must reproduce the published
# 1.1832 / 0.8827 to four digits, or the --cot-scale patch has changed the K=1 path and no
# scaled arm below it can be read.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_rebase" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
B=$REF_BUNDLE
C="$REF_CAP_LADDER"
for f in perf/of3t_gradients/instrument_a_bundle.py; do
  [ -f "$W/$f" ] || { echo "MISSING SCRIPT: $f -- this checkout is stale"; exit 2; }
done
grep -q -- "--cot-scale" "$W/perf/of3t_gradients/instrument_a_bundle.py" || {
  echo "STALE INSTRUMENT: instrument_a_bundle.py has no --cot-scale, ship the patched copy"; exit 2; }
[ -f "$C/block47_boundary.pt" ] || { echo "no block-47 capture at $C"; exit 2; }
# THE SCALES MUST NOT ALL BE POWERS OF TWO, and this script first shipped as if they could be.
# Scaling every input to a linear computation by 2^n shifts exponents and leaves every mantissa
# and every rounding decision untouched, so a power-of-two arm is invariant BY THE ARITHMETIC,
# whatever the defect is. The first run used 1, 1/8 and 1/64, got bit-identical factors, and that
# result was vacuous: it could not have come out any other way. The non-power-of-two arms are the
# ones with teeth -- they re-round the cotangent into bf16 differently, so the whole backward
# sees genuinely different mantissas. The powers of two are kept as an arithmetic control: they
# MUST read bit-identical, and if they ever do not, the harness is not linear in the cotangent
# and nothing else here can be read.
for k in 1.0 0.125 0.015625 0.1 0.0137; do
  echo "=== block 47, cot-scale $k  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py --block 47 --crop 64 \
      --transpose-bias shipped --scale-pair-bias off --cot-scale "$k" \
      --tag "COTSCALE_k${k}" \
      --bundle "$B" --manifest-json "$B/MANIFEST.json" --cap "$C" \
      --out-dir perf/of3t_rebase \
      --capture-report perf/of3t_rebase/capture_trunk_boundary_043_ladder.json 2>&1 \
    | grep -E "D37 cotangent|^median |track" | sed "s/^/  /"
  echo "  arm exit $?"
done
echo "COTSCALE_ALLDONE $(date -u +%FT%TZ)"
