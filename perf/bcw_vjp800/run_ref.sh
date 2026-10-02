#!/usr/bin/env bash
# The card-free half of the 800-token gradient grade: build the float64 references.
#
# Three legs, in order, because the cheap ones are the evidence that the expensive one is
# measuring what it claims:
#   1. n=288 chunked      the paired baseline, same harness and same seeding as 800
#   2. n=288 unchunked    the same thing with lowmem off (--floor 288), so the chunking's
#                         effect on the reading can be quoted at a real size, not just at 128
#   3. n=800 chunked      the grade's reference
#
# CPU only: `.venvs/jfp` has torch and no ttnn, so this cannot open a chip even by accident.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=/home/moritz/.venvs/jfp/bin/python3
OUT=perf/bcw_vjp800/out
mkdir -p "$OUT"
export OMP_NUM_THREADS=${THREADS:-8}
T=${THREADS:-8}

run() {
  echo "=== $* :: $(date -u +%FT%TZ)"
  "$PY" perf/bcw_vjp800/vjp800.py ref --threads "$T" "$@"
  echo "=== rc=$? :: $(date -u +%FT%TZ)"
}

run --n 288 --cache perf/bcw_vjp800/cache_n288
run --n 288 --floor 288 --cache perf/bcw_vjp800/cache_n288_nochunk
"$PY" perf/bcw_vjp800/chunk_delta.py perf/bcw_vjp800/cache_n288 \
      perf/bcw_vjp800/cache_n288_nochunk
run --n 800 --cache perf/bcw_vjp800/cache_n800
echo "=== ALL DONE :: $(date -u +%FT%TZ)"
