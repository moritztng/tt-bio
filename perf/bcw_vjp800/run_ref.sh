#!/usr/bin/env bash
# The card-free half of the 800-token gradient grade: build a float64 reference at n.
#   run_ref.sh <n> <cache> [extra vjp800 ref args...]
#
# One process per stage. The cotangent pass holds eight float64 boundaries, two retained
# boundary grads and one block's residual stream at once, and that set is ~n^2: at 288 it was
# 1.77 GB on top of the 2.01 GB model, so 800 is ~15.7 GB. Each stage exiting hands its arenas
# back, and MALLOC_ARENA_MAX caps what glibc keeps while it runs -- without it the same 288
# cotangent pass held 10.91 GB against 3.78 GB with it, 2.9x, all of it arena retention.
#
# CPU only: `.venvs/jfp` has torch and no ttnn, so this cannot open a chip even by accident.
set -u
cd "$(dirname "$0")/../.." || exit 1
N=${1:?usage: run_ref.sh <n> <cache> [args...]}
CACHE=${2:?usage: run_ref.sh <n> <cache> [args...]}
shift 2
PY=/home/moritz/.venvs/jfp/bin/python3
T=${THREADS:-8}
export OMP_NUM_THREADS=$T MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2} \
       MALLOC_TRIM_THRESHOLD_=134217728
BLOCKS=${BLOCKS:-0,3,7}

stage() {
  echo "=== n=$N $* :: $(date -u +%FT%TZ)"
  "$PY" perf/bcw_vjp800/vjp800.py ref --n "$N" --cache "$CACHE" --threads "$T" \
        --blocks "$BLOCKS" "$@"
  local rc=$?
  echo "=== rc=$rc :: $(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || exit "$rc"
}

stage --stage cot "$@"
for j in ${BLOCKS//,/ }; do stage --stage arms --block "$j" "$@"; done
stage --stage fin "$@"
echo "=== ALL DONE n=$N :: $(date -u +%FT%TZ)"
