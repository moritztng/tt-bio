#!/usr/bin/env bash
# A18's forward discriminator and, only if it clears, the diffusion-scope gradient (D23 #4).
#
# A18's first clause is binding: a disagreeing forward invalidates the gradient comparison taken
# at it. The discriminator read 1.114e-01 against the 0.5.0 boundary and D21 withheld the
# gradient on that basis. This re-reads it against the 0.4.3 boundary, which reproduces its own
# bundle at worst 0.000e+00 over 761 tensors covering 89.21 % of the squared gradient norm.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_tape" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
echo "=== device gradient at 0.4.3, structs 0  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/device_gradient.py --structs 0 --tag _043 \
    --cap "$REF_DIFFCAP" --out-dir perf/of3t_rebase
echo "=== device gradient exit $? $(date -u +%FT%TZ) ==="
echo "DEVGRAD_ALLDONE $(date -u +%FT%TZ)"
