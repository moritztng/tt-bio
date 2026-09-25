#!/usr/bin/env bash
# One floor arm per invocation, CPU only -- no card, no lease.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-condtrans
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t_gradients/of3pkg:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=${OMP:-12}
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/tmp/of3t/condtrans
mkdir -p "$O" "$W/perf/of3t_condtrans"
P=$1
echo "=== floor $P $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_condtrans/floor_bf16.py --policy "$P" --expect-version 0.5.0 \
  --out "$O/floor_${P}.pt" --report "$W/perf/of3t_condtrans/FLOOR_${P}.json"
echo "=== floor $P exit $? $(date -u +%FT%TZ) ==="
