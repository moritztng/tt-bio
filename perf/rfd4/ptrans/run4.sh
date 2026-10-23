#!/bin/bash
# run4: the shipped change (O alone, T1 reverted) on the merged tree against origin/main, same pass_bench protocol as run3.
#   bash perf/rfd4/ptrans/run4.sh <chip> <outdir> [rounds]     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}"); ROUNDS=${3:-2}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
cd "$WT"
declare -A REV=([main]=$(git rev-parse origin/main) [o]=$(git rev-parse HEAD))
for arm in main o; do mkdir -p "$R/$arm"; git archive "${REV[$arm]}" | tar -x -C "$R/$arm"; done
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-ptrans TT_BIO_TRACE_REGION_MB=1536
stamp "start chip $N main ${REV[main]} o ${REV[o]} py $PY"
for n in ${SIZES:-512 256}; do
  for r in $(seq 1 "$ROUNDS"); do
    for arm in main o; do
      (cd "$R/$arm" && PYTHONPATH=$R/$arm timeout 1800 "$PY" "$WT/perf/rfd4/e2e/pass_bench.py" \
        --weights "$HOME/rfd4-bundle" --n "$n" --reps 30 --out "$R/$n-$arm$r.json" --dump "$R/$n-$arm$r.pt" \
        > "$R/$n-$arm$r.log" 2>&1)
      stamp "$n-$arm$r rc=$? $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['tt_bio'], d['call_traced_ms'], d['aiclk'])" "$R/$n-$arm$r.json" 2>/dev/null)"
    done
    "$PY" - "$R/$n-main$r.pt" "$R/$n-o$r.pt" >> "$R/done" 2>&1 <<'PY'
import os, sys, torch
a = torch.load(sys.argv[1])
for p in sys.argv[2:]:
    b = torch.load(p)
    print("identity", p, {k: bool(torch.equal(a[k], b[k])) for k in a})
for p in sys.argv[1:]:
    os.remove(p)
PY
  done
done
stamp end
