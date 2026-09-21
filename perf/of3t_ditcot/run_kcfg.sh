#!/usr/bin/env bash
# One KCFG arm per invocation, fresh process, one device context, card 0.
# RENORM is the composition default (autograd.py:91 env_flag(..., True)) -- the flag is
# deliberately NOT set here, per AMENDMENT 4.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-ditcot
cd "$W" || exit 1
source /home/ttuser/tt-bio-dev/env/bin/activate
export OMP_NUM_THREADS=${OMP:-4}
O=/tmp/of3t/ditcot
mkdir -p "$O" "$W/perf/of3t_ditcot"
TAG=$1; ARM=$2; shift 2
echo "=== kcfg $TAG arm=$ARM $(date -u +%FT%TZ) ==="
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-ditcot \
timeout 3000 python3 perf/of3t_ditcot/kcfg_pull.py --arm "$ARM" \
  --fire-out "$W/perf/of3t_ditcot/FIRE_${TAG}.json" \
  --dump-per-tensor --dump-grads "$O/kcfg_${TAG}.pt" \
  --out-dir "$W/perf/of3t_ditcot" --tag "_${TAG}" "$@"
echo "=== kcfg $TAG exit $? $(date -u +%FT%TZ) ==="
