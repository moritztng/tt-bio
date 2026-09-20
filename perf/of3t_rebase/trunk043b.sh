#!/usr/bin/env bash
# The six crop-64 arms, re-run at scale_pair_bias=off to match the arms they replace.
#
# Audited rather than assumed: all six published crop-64 arms recorded
# `shipped_config.scale_pair_bias = false`, and only the three 48-block stack arms recorded
# `true`. The first attempt pinned `on` for both, which moved the bias convention and the
# upstream revision together on the ladder -- the exact two-moving-parts defect this row exists
# to undo. The stack arms keep `on`; the ladder gets `off`.
set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_rebase:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/of3t_rebase/bundle_min_043
COMMON=(--bundle "$B" --manifest-json "$B/MANIFEST.json" --cap /home/ttuser/of3t_rebase/cap043
        --out-dir perf/of3t_rebase --scale-pair-bias off
        --capture-report perf/of3t_rebase/capture_trunk_boundary_043.json)
for blk in 0 23 47; do
  for arm in shipped off; do
    echo "=== block $blk crop64 tb-$arm spb-off  $(date -u +%FT%TZ) ==="
    "$PY" perf/of3t_gradients/instrument_a_bundle.py --block "$blk" --crop 64 \
        --transpose-bias "$arm" --tag "043spb_block${blk}_crop64_tb${arm}" "${COMMON[@]}"
    echo "=== block $blk tb-$arm exit $? ==="
  done
done
echo "TRUNKB_ALLDONE $(date -u +%FT%TZ)"
