#!/usr/bin/env bash
# of3t-cond043: upstream 0.4.3's own bf16 floor, at the 0.4.3 capture. CPU only, no card, no lease.
#
# The three things that make this a different measurement from perf/of3t_condtrans/run_floor.sh:
#   PYTHONPATH resolves openfold3 to /home/ttuser/of3t_refprec/of3pkg043, which is 0.4.3;
#   --cap is /home/ttuser/of3t_softgrad/diffcap043, the 0.4.3 capture, whose sub_boundary.pt
#   carries the 0.4.3 float64 reference every rel here divides by;
#   OMP_NUM_THREADS is 8, the value recap043.sh built that capture with. A float64 CPU reduction
#   is a function of its thread count, so every arm meant to be differenced uses the same 8.
#
# of3t_gradients/pylibs is DELIBERATELY not on the path: it holds an openfold3-0.5.0.dist-info,
# and a 0.5.0 dist-info behind a 0.4.3 source tree is exactly what makes importlib.metadata report
# the wrong version. floor_bf16.py reads the version off the imported module's own directory and
# --expect-version 0.4.3 hard-fails if they disagree, but the path stays clean anyway.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-cond043
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t_refprec/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
CAP=${3:-/home/ttuser/of3t_softgrad/diffcap043}
O=/tmp/of3t/cond043
mkdir -p "$O"
P=$1
TAG=${2:-$1}
echo "=== floor043 $P tag $TAG cap $CAP OMP=$OMP_NUM_THREADS $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_condtrans/floor_bf16.py --policy "$P" --expect-version 0.4.3 --cap "$CAP" \
  --out "$O/floor043_${TAG}.pt" --report "$W/perf/of3t_cond043/FLOOR043_${TAG}.json"
echo "=== floor043 $TAG exit $? $(date -u +%FT%TZ) ==="
