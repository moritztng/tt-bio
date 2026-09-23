#!/usr/bin/env bash
# of3t-msaamp: one upstream 0.4.3 arm at the msa_module boundary. CPU only, no card, no lease.
# OMP_NUM_THREADS=8 as in of3t-tapeamp's arms.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msaamp
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t-campaign-refs/of3pkg043:/home/ttuser/of3t_frame384/ref:/home/ttuser/of3t_hostleg/deps:/home/ttuser/of3t_hostleg/pylibs:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
BND=${BND:-/home/ttuser/of3t_msaamp/cap043b/boundary_msa_module.pt}
POLICY=$1; TAG=$2; shift 2
echo "=== msa_amp_arm $POLICY tag $TAG OMP=$OMP_NUM_THREADS host=$(hostname) load[$(cut -d' ' -f1-3 /proc/loadavg)] $(date -u +%FT%TZ) ==="
nice -n 5 "$PY" perf/of3t_msaamp/msa_amp_arm.py --policy "$POLICY" --expect-version 0.4.3 \
  --boundary "$BND" --tag "$TAG" --dump "/home/ttuser/of3t_msaamp/grads_$TAG.pt" "$@" \
  --report "$W/perf/of3t_msaamp/AMP_$TAG.json"
echo "=== msa_amp_arm $TAG exit $? $(date -u +%FT%TZ) ==="
