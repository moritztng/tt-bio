#!/usr/bin/env bash
# of3t-tapeamp: upstream 0.4.3's own forward AND gradient at the diffusion boundary. CPU only,
# no card, no lease. OMP_NUM_THREADS=8 is the value recap043.sh built the capture with and the
# value every arm on record used -- a float64 CPU reduction is a function of its thread count,
# so an arm meant to be differenced against those uses the same 8.
#
# of3t_gradients/ does not exist on qb1, so bundle_min comes from of3t_frame384/ref and the
# capture from of3t_hostleg/diffcap043. Both are checked in-process: --expect-version reads the
# version off the resolved tree, and the fingerprint guard checks the capture's grad_f64 keys
# against this tree's DiffusionModule parameter names.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-tapeamp
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t-campaign-refs/of3pkg043:/home/ttuser/of3t_frame384/ref:/home/ttuser/of3t_hostleg/deps:/home/ttuser/of3t_hostleg/pylibs:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
CAP=/home/ttuser/of3t_hostleg/diffcap043
MATCH="$W/perf/of3t_ditref/device_gradient_r043_ours_per_tensor.json"
POLICY=$1
TAG=${2:-$1}
EXTRA=${3:-}
echo "=== amp_arm $POLICY tag $TAG cap $CAP OMP=$OMP_NUM_THREADS host=$(hostname) $(date -u +%FT%TZ) ==="
nice -n 5 "$PY" perf/of3t_tapeamp/amp_arm.py --policy "$POLICY" --expect-version 0.4.3 \
  --cap "$CAP" --tag "$TAG" --match-scope "$MATCH" $EXTRA \
  --report "$W/perf/of3t_tapeamp/AMP_${TAG}.json"
echo "=== amp_arm $TAG exit $? $(date -u +%FT%TZ) ==="
