#!/usr/bin/env bash
# Both diffusion-loop measurements, one after the other, on one card.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES="${CARD:-1}" TT_BIO_LEASE_CARDS="${CARD:-1},3"
export TT_BIO_LEASE_HOLDER=worker:b2z2-diffusion-loop-attack
echo "=== loop_host_split $(date -u +%H:%M:%S) ==="
timeout 2400 "$PY" perf/b2z2_diff/loop_host_split.py --out perf/b2z2_diff/loop_host_split_qb2c${CARD:-1}.json
echo "=== trace_ab $(date -u +%H:%M:%S) ==="
timeout 2400 "$PY" perf/bioir_dispatch/trace_ab_boltz2_512.py --out perf/b2z2_diff/trace_ab_qb2c${CARD:-1}.json --reps 3 --skip-census
echo "=== done $(date -u +%H:%M:%S) ==="
