#!/bin/bash
# run4: the one-hot pair gather (+ levers R, E) against main at 512 on pdl1.
#   1. the pair device test on the branch
#   2. host_split ABAB, main / branch / main / branch, 4 designs each: wall and device split per design
#   3. the quality grade: design.py seeds 0-16, rng upstream, main then branch (seed 0 is the compile run)
#   bash perf/rfd4/glue/run4.sh <chip> <outdir>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}")
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
BASE=${BASE:-6e42015fc}
SPEC=$HOME/rfd4-inputs/pdl1-512.yaml W=$HOME/rfd4-bundle
mkdir -p "$R/main" "$R/design"; cd "$WT"
git archive "$BASE" | tar -x -C "$R/main"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue
stamp "start chip $N branch $(git rev-parse --short HEAD) dirty=$(git status --short tt_bio | wc -l) base $BASE"
PYTHONPATH=$WT timeout 1200 "$PY" -m pytest -x -q -s -m device tests/test_rfd4_pair_device.py > "$R/pairtest.log" 2>&1
stamp "pairtest rc=$? $(tail -1 "$R/pairtest.log")"
tree() { if [ "$1" = main ]; then echo "$R/main"; else echo "$WT"; fi; }
for r in 1 2; do
  for arm in main branch; do
    t=$(tree $arm)
    (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$WT/perf/rfd4/e2e/host_split.py" --spec "$SPEC" --weights "$W" \
      --out "$R/split-$arm$r" --designs 4 > "$R/split-$arm$r.log" 2>&1)
    stamp "split-$arm$r rc=$?"
  done
done
for arm in main branch; do
  t=$(tree $arm)
  (cd "$t" && PYTHONPATH=$t timeout 5400 "$PY" "$WT/perf/rfd4/e2e/design.py" --spec "$SPEC" --weights "$W" \
    --out "$R/design/$arm" --rng upstream --seeds $(seq 0 16) > "$R/design/$arm.log" 2>&1)
  stamp "design-$arm rc=$? warm=$(grep -c '"kind": "warm"' "$R/design/$arm/runs.jsonl" 2>/dev/null)"
done
stamp end
