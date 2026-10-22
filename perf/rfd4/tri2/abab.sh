#!/bin/bash
# ABAB of rfd4-perf-tri2 against its base on ONE card in ONE session: warm designs base, branch, base, branch
# at each size (perf/rfd4/e2e/design.py, AICLK sampled inside every design), then the CIFs compared.
#   perf/rfd4/tri2/abab.sh <card> <base-tree> <out-dir> [sizes...]     (default sizes: 128 512)
set -u
card=$1 btree=$2 out=$3; shift 3
sizes=${*:-128 512}
tree=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri2
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"; }
stamp "start branch=$(git -C "$tree" log --oneline -1 | cut -c1-60) base=$(git -C "$btree" log --oneline -1 | cut -c1-60)"
leg() {  # leg <name> <tree> <n> <designs>
    (cd "$2" && PYTHONPATH=$PWD timeout 2400 "$py" perf/rfd4/e2e/design.py \
        --spec "$HOME/rfd4-inputs/pdl1-$3.yaml" --weights "$weights" --out "$out/$3/$1" --designs "$4") \
        >"$out/$3/$1.log" 2>&1
    stamp "n=$3 $1 rc=$? $(tail -1 "$out/$3/$1/runs.jsonl" 2>/dev/null | cut -c1-300)"
}
for n in $sizes; do
    mkdir -p "$out/$n"
    d=$([ "$n" -le 128 ] && echo 3 || echo 2)
    leg base1 "$btree" "$n" "$d"; leg branch1 "$tree" "$n" "$d"
    leg base2 "$btree" "$n" "$d"; leg branch2 "$tree" "$n" "$d"
done
stamp end
