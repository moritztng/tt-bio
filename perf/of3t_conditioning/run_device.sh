#!/usr/bin/env bash
# Instrument A at conditioning scope. A18's forward discriminator runs first; pass
# --forward-only to stop there.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="$W/perf/of3t_tape:$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CARD=${CARD:-2}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-conditioning
PY=/home/ttuser/tt-bio-dev/env/bin/python
exec "$PY" perf/of3t_conditioning/device_cond_gradient.py "$@"
