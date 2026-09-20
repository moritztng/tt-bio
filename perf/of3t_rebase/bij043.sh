#!/usr/bin/env bash
# Amendment 3: re-derive the bijection against the 0.4.3 model, 4,170 parameters.
#
# The campaign's reach figures are all denominated in 4,147 -- the 0.5.0 model's parameter count.
# The 0.4.3 model has 4,170: it carries 24 per-block DiT layer_norm_z and not the 1 hoisted
# shared one. Our device model builds exactly those per-block norms when the checkpoint has no
# shared one, so coverage should go UP rather than down. This measures it instead of asserting.
set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/of3t_rebase/bundle_min_043
echo "=== bijection against BUNDLE-MIN-043  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/instrument_full_model.py --materialise 64 --tag of3_full_043 \
    --bundle "$B" --manifest-json "$B/MANIFEST.json" \
    --presence-file grad_presence_043.json --out-dir perf/of3t_rebase
echo "=== bijection exit $? $(date -u +%FT%TZ) ==="
