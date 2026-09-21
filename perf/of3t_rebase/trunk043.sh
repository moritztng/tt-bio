#!/usr/bin/env bash
# The six trunk arms against BUNDLE-MIN-043, plus the 48-block stack in both arms.
#
# Amendment 2a: blocks 0, 23 and 47 at crop 64 are what carry the information. The 48-block
# stack is saturated -- both arms sit above the zero model's 1.0 -- so it is reported but it is
# not the cut.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_rebase" "$W")"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
B=$REF_BUNDLE
# --scale-pair-bias on, explicitly. wk/of3t flipped the OF3 trunk default from True to False
# during this row's passes (openfold3_trunk.py:154, of3t-pairbias' "land the mechanism, do not
# flip the default"), and the arms being replaced were taken at True. `shipped` no longer means
# what it meant when they were recorded, so the convention is pinned rather than inherited.
COMMON=(--bundle "$B" --manifest-json "$B/MANIFEST.json" --cap "$REF_CAP"
        --out-dir perf/of3t_rebase --scale-pair-bias on
        --capture-report perf/of3t_rebase/capture_trunk_boundary_043.json)

for blk in 0 23 47; do
  for arm in shipped off; do
    echo "=== block $blk crop64 tb-$arm  $(date -u +%FT%TZ) ==="
    "$PY" perf/of3t_gradients/instrument_a_bundle.py --block "$blk" --crop 64 \
        --transpose-bias "$arm" --tag "043_block${blk}_crop64_tb${arm}" "${COMMON[@]}"
    echo "=== block $blk tb-$arm exit $? ==="
  done
done

for arm in shipped off; do
  echo "=== stack 0..47 n384 tb-$arm  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py --block 0 --stack 48 \
      --transpose-bias "$arm" --tag "043_stack0_47_n384_tb${arm}" "${COMMON[@]}"
  echo "=== stack tb-$arm exit $? ==="
done
echo "TRUNK_ALLDONE $(date -u +%FT%TZ)"
