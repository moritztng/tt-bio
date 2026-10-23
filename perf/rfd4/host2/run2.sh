#!/bin/bash
# run2: lever P (the pair build as one program) against main, at 512 and 128 on pdl1.
#   Both arms run from git-archive snapshots, so neither carries the worktree (run1 saw a tree effect at 128).
#   1. the pair device test on the branch snapshot
#   2. host_split ABAB per size, main / branch / main / branch, 4 designs a leg (3 warm)
#   3. the grade at 512: design.py seeds 0-31 per arm, then grade_seeds (paired CA RMSD vs main, seed floor)
#      and the oracle-free binder distribution grade
#   bash perf/rfd4/host2/run2.sh <chip> <outdir>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}")
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
BASE=${BASE:-$(git -C "$WT" rev-parse origin/main)}
HEAD_SHA=$(git -C "$WT" rev-parse HEAD)
W=$HOME/rfd4-bundle SPEC=$HOME/rfd4-inputs/pdl1-512.yaml
mkdir -p "$R/main" "$R/branch" "$R/design"; cd "$WT"
git archive "$BASE" | tar -x -C "$R/main"
git archive "$HEAD_SHA" | tar -x -C "$R/branch"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-host2
stamp "start chip $N branch $HEAD_SHA base $BASE"
(cd "$R/branch" && RFD4_SPEC=$SPEC PYTHONPATH=$R/branch timeout 1200 "$PY" -m pytest -q -s -m device \
  tests/test_rfd4_pair_device.py > "$R/pairtest.log" 2>&1)
stamp "pairtest rc=$? $(tail -1 "$R/pairtest.log")"
for sz in 512 128; do
  for r in 1 2; do
    for arm in main branch; do
      t=$R/$arm
      (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$t/perf/rfd4/e2e/host_split.py" \
        --spec "$HOME/rfd4-inputs/pdl1-$sz.yaml" --weights "$W" \
        --out "$R/s$sz-$arm$r" --designs 4 > "$R/s$sz-$arm$r.log" 2>&1)
      stamp "s$sz-$arm$r rc=$?"
    done
  done
done
for arm in main branch; do
  t=$R/$arm
  (cd "$t" && PYTHONPATH=$t timeout 5400 "$PY" "$t/perf/rfd4/e2e/design.py" --spec "$SPEC" --weights "$W" \
    --out "$R/design/$arm" --rng upstream --seeds $(seq 0 31) > "$R/design/$arm.log" 2>&1)
  stamp "design-$arm rc=$?"
done
exec 9>&-
cd "$R/branch"
PYTHONPATH=$R/branch "$PY" perf/rfd4/bfp8/grade_seeds.py --base "$R/design/main" --arm "$R/design/branch" \
  --out "$R/grade32.json" > "$R/grade32.log" 2>&1
stamp "grade_seeds rc=$?"
mkdir -p "$R/q/main" "$R/q/branch"
for a in main branch; do ln -sf "$R/design/$a"/*_generated.cif "$R/q/$a/"; done
cp /home/ttuser/.coworker/state/rfd4/QUALITY.json "$R/quality512.json" 2>/dev/null
PYTHONPATH=$R/branch "$PY" perf/rfd4/quality.py grade --feature binder --tt "$R/q/branch" --ref "$R/q/main" \
  --allow-no-oracle --label "pair_build (P) vs main, 512-token PD-L1 binder, seeds 0-31" \
  --out "$R/quality512.json" > "$R/quality512.log" 2>&1
stamp "quality rc=$?"
stamp end
