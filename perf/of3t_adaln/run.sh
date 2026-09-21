#!/usr/bin/env bash
# The AdaLN micro-arm. One card, minutes.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
export PYTHONPATH="$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
CARD=${CARD:-3}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-adaln
exec /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_adaln/adaln_micro.py "$@"
