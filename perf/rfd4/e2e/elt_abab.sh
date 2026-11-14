#!/bin/bash
# The elementwise bet on one card in one session: device profile of the branch, device suites, then
# ABAB warm designs in the same tree with TT_BIO_RESIDUAL_DUAL=0 (base: typecast + add + typecast)
# and =1 (branch). AICLK is sampled inside every design (design.py).
#   perf/rfd4/e2e/elt_abab.sh <card> <out-dir> [rounds] [designs per leg, the first is the compile] [prof 1|0]
set -u
card=$1 out=$2 rounds=${3:-2} designs=${4:-2} prof=${5:-1}
wt=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
spec=${RFD4_SPEC:-$HOME/rfd4-inputs/pdl1-128.yaml}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rfd4-perf-elt
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
cd "$wt"
stamp "start $(git log --oneline -1)"
if [ "$prof" = 1 ]; then
    bash perf/rfd4/e2e/elt_prof.sh "$out/prof" "$card" > "$out/prof.log" 2>&1
    stamp "prof rc=$?"
fi
PYTHONPATH=$wt timeout 1800 "$py" -m pytest -q -s -m device tests/test_rfd4_trunk_parity.py \
    tests/test_rfd4_trunk_golden.py > "$out/suites.log" 2>&1
stamp "suites rc=$? $(tail -1 "$out/suites.log")"
leg() {  # leg <name> <dual>
    TT_BIO_RESIDUAL_DUAL=$2 PYTHONPATH=$wt timeout 1200 "$py" perf/rfd4/e2e/design.py --spec "$spec" \
        --weights "$weights" --out "$out/$1" --designs "$designs" > "$out/$1.log" 2>&1
    stamp "design $1 rc=$? $(tail -1 "$out/$1/runs.jsonl" 2>/dev/null)"
}
for r in $(seq 1 "$rounds"); do
    leg "base$r" 0
    leg "branch$r" 1
done
stamp end
