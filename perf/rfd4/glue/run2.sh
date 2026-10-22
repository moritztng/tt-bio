#!/bin/bash
# run2: levers R (recycle as one dual write) + E (no trunk-entry copy) vs main, depth-3 trunk call at 512 and 256.
# pass_bench.py per tree, interleaved main/branch, 30 pass reps (10 call reps) each, call outputs dumped for identity.
#   bash perf/rfd4/glue/run2.sh <chip> <outdir> <rounds>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}"); ROUNDS=${3:-2}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
BASE=${BASE:-6e42015fc}
mkdir -p "$R/main"; cd "$WT"
git archive "$BASE" | tar -x -C "$R/main"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue TT_BIO_TRACE_REGION_MB=1536
stamp "start chip $N branch $(git rev-parse --short HEAD) dirty=$(git status --short tt_bio | wc -l) base $BASE"
for n in ${SIZES:-512 256}; do
  for r in $(seq 1 "$ROUNDS"); do
    for arm in main branch; do
      tree=$WT; [ $arm = main ] && tree=$R/main
      (cd "$tree" && PYTHONPATH=$tree timeout 1800 "$PY" "$WT/perf/rfd4/e2e/pass_bench.py" --weights "$HOME/rfd4-bundle" \
        --n "$n" --reps 30 --out "$R/$n-$arm$r.json" --dump "$R/$n-$arm$r.pt" > "$R/$n-$arm$r.log" 2>&1)
      stamp "$n-$arm$r rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['tt_bio'], d['call_eager_ms'], d.get('call_traced_ms'), d['aiclk'])" "$R/$n-$arm$r.json" 2>/dev/null)"
    done
    "$PY" - "$R/$n-main$r.pt" "$R/$n-branch$r.pt" >> "$R/done" 2>&1 <<'PY'
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
print("identity", sys.argv[2], {k: bool(torch.equal(a[k], b[k])) for k in a})
PY
    rm -f "$R/$n-main$r.pt" "$R/$n-branch$r.pt"
  done
done
stamp end
