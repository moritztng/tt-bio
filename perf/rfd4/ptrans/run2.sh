#!/bin/bash
# run2: run1 measured both standalone arms on the branch (the scripts put the worktree ahead of PYTHONPATH). This
# repeats orz.py / relu.py per tree with that fixed, then host_split ABAB at 512 (4 designs a leg) for s/design.
#   bash perf/rfd4/ptrans/run2.sh <chip> <outdir> [rounds]     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}"); ROUNDS=${3:-2}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
BASE=${BASE:-$(cd "$WT" && git merge-base HEAD origin/main)}
SPEC=$HOME/rfd4-inputs/pdl1-512.yaml W=$HOME/rfd4-bundle
mkdir -p "$R/main"; cd "$WT"
git archive "$BASE" | tar -x -C "$R/main"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-ptrans TT_BIO_TRACE_REGION_MB=1536
stamp "start chip $N branch $(git rev-parse --short HEAD) dirty=$(git status --short tt_bio | wc -l) base $BASE py $PY"
tree() { if [ "$1" = main ]; then echo "$R/main"; else echo "$WT"; fi; }
for r in $(seq 1 "$ROUNDS"); do
  for arm in main branch; do
    t=$(tree $arm)
    for s in orz relu; do
      x=; [ $s = relu ] && x=--attr
      (cd "$t" && PYTHONPATH=$t timeout 900 "$PY" "$WT/perf/rfd4/ptrans/$s.py" $x --out "$R/$s-$arm$r.json" \
        > "$R/$s-$arm$r.log" 2>&1)
      stamp "$s-$arm$r rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"tt_bio\"], [(r.get(\"shape\", r.get(\"n\")), r.get(\"arm\"), r[\"ms\"]) for r in d[\"rows\"]][:4], d[\"aiclk\"])" "$R/$s-$arm$r.json" 2>/dev/null)"
    done
  done
done
for r in 1 2; do
  for arm in main branch; do
    t=$(tree $arm)
    (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$WT/perf/rfd4/e2e/host_split.py" --spec "$SPEC" --weights "$W" \
      --out "$R/split-$arm$r" --designs 4 > "$R/split-$arm$r.log" 2>&1)
    stamp "split-$arm$r rc=$?"
  done
done
stamp end
