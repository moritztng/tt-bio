#!/bin/bash
# run1: lever S (the pair scale shipped as whole rows) against main, at 512 and 128 on pdl1.
#   1. the pair device test on the branch, with the scale-column exactness test
#   2. host_split ABAB per size, main / branch / main / branch, 4 designs a leg (3 warm)
#   3. one more profiled branch design at 512 (cProfile top, attributes the CIF write and the glue)
#   bash perf/rfd4/host2/run1.sh <chip> <outdir>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}")
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
BASE=${BASE:-$(git -C "$WT" merge-base HEAD origin/main)}
W=$HOME/rfd4-bundle
mkdir -p "$R/main"; cd "$WT"
git archive "$BASE" | tar -x -C "$R/main"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-host2
stamp "start chip $N branch $(git rev-parse --short HEAD) dirty=$(git status --short tt_bio | wc -l) base $BASE"
RFD4_SPEC=$HOME/rfd4-inputs/pdl1-512.yaml PYTHONPATH=$WT timeout 1200 "$PY" -m pytest -q -s -m device \
  tests/test_rfd4_pair_device.py > "$R/pairtest.log" 2>&1
stamp "pairtest rc=$? $(tail -1 "$R/pairtest.log")"
tree() { if [ "$1" = main ]; then echo "$R/main"; else echo "$WT"; fi; }
for sz in 512 128; do
  for r in 1 2; do
    for arm in main branch; do
      t=$(tree $arm)
      (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$WT/perf/rfd4/e2e/host_split.py" \
        --spec "$HOME/rfd4-inputs/pdl1-$sz.yaml" --weights "$W" \
        --out "$R/s$sz-$arm$r" --designs 4 > "$R/s$sz-$arm$r.log" 2>&1)
      stamp "s$sz-$arm$r rc=$?"
    done
  done
done
(PYTHONPATH=$WT timeout 2400 "$PY" "$WT/perf/rfd4/e2e/host_split.py" --spec "$HOME/rfd4-inputs/pdl1-512.yaml" \
  --weights "$W" --out "$R/prof512" --designs 2 --profile > "$R/prof512.log" 2>&1)
stamp "prof512 rc=$?"
stamp end
