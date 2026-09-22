#!/usr/bin/env bash
# of3t-cotterm: the ACROSS attribution, from the captures the arms produced.
#   runsplit.sh <DEVTAG> [--tree]
# CPU only, no board: both operands are already on the host.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-cotterm
O=/tmp/of3t/of3t-cotterm
D=/home/ttuser/of3t_frame384
cd "$W"
TAG=$1; shift
TREEARG=""
if [ "${1:-}" = "--tree" ]; then TREEARG="--tree $D/of3pkg043"; export PYTHONPATH=$D/ref:$D/deps; fi
echo "=== split $TAG start $(date -u +%FT%TZ) host=$(hostname) ==="
OMP_NUM_THREADS=16 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_cotterm/split.py \
  --ref-apb "$O/apb_ref_n384.pt" --dev-apb "$O/apb_${TAG}_n384.pt" \
  --ref-ln "$O/ln_ref_n384.pt"  --dev-ln  "$O/ln_${TAG}_n384.pt" \
  --boundary "$D/boundary_n384.pt" --validate-block 44 $TREEARG \
  --out "$W/perf/of3t_cotterm/SPLIT_${TAG}_N384.json" \
  --cf-out "$W/perf/of3t_cotterm/CF_${TAG}_N384.json"
echo "=== split exit $? $(date -u +%FT%TZ) ==="
