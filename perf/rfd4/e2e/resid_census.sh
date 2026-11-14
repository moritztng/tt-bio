#!/bin/bash
# Census v2 at the gating sizes: elt_prof.sh + census_v2.py at each n, under the chip flock (D-orch-40).
#   perf/rfd4/e2e/resid_census.sh <out-dir> <card> [n ...]     (default n: 256 512)
set -u
out=$1 card=$2; shift 2
ns=${*:-256 512}
wt=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
mkdir -p "$out"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$card.lock"
stamp "wait flock chip$card"
flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$card" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git -C "$wt" log --oneline -1)"
for n in $ns; do
    bash "$wt/perf/rfd4/e2e/elt_prof.sh" "$out/prof-$n" "$card" --n "$n" > "$out/prof-$n.log" 2>&1
    stamp "prof $n rc=$?"
    PYTHONPATH=$wt "$py" "$wt/perf/rfd4/e2e/census_v2.py" "$out/prof-$n" "$out/e2e-$n-v2.json" >> "$out/prof-$n.log" 2>&1
    stamp "census $n rc=$?"
done
stamp end
