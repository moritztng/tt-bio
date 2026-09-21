#!/usr/bin/env bash
# One KCFG arm on the TRUNK scope -- the only scope that reaches T1 and T3.
# RENORM is the composition default; the flag is deliberately not set (AMENDMENT 4).
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-ditcot
cd "$W" || exit 1
source /home/ttuser/tt-bio-dev/env/bin/activate
export OMP_NUM_THREADS=${OMP:-4}
O=/tmp/of3t/ditcot
mkdir -p "$O" "$W/perf/of3t_ditcot"
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
TAG=$1; ARM=$2; shift 2
echo "=== kcfg-trunk $TAG arm=$ARM $(date -u +%FT%TZ) ==="
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-ditcot \
timeout 3000 python3 perf/of3t_ditcot/kcfg_pull.py --kcfg-arm "$ARM" --harness trunk \
  --fire-out "$W/perf/of3t_ditcot/FIRE_trunk_${TAG}.json" \
  --boundary "$B" --cap-last "$C" --arm shipped \
  --out "$O/trunk_${TAG}.pt" \
  --report "$W/perf/of3t_ditcot/TRUNK_${TAG}.json" "$@"
echo "=== kcfg-trunk $TAG exit $? $(date -u +%FT%TZ) ==="
