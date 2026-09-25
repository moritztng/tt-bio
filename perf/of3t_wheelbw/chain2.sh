#!/bin/bash
# Re-run the VJP after chain.sh, with the relu kink fix and the two withdrawals recorded.
set -u
WT=/home/moritz/.coworker/wt/of3t-wheelbw
OUT=$WT/perf/of3t_wheelbw/out
PY=/home/moritz/tt-bio/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-wheelbw
while kill -0 "${1:-0}" 2>/dev/null; do sleep 20; done
for dt in float32 bfloat16; do
  echo "=== vjp $dt (rerun) $(date -u +%FT%TZ) ==="
  timeout 2400 $PY perf/of3t_wheelbw/vjp.py --dtype $dt --out "$OUT/vjp_${dt}_r2.json" 2>&1 \
    | grep -v 'DEBUG *|\|Config{'
done
echo "CHAIN2 DONE $(date -u +%FT%TZ)"
