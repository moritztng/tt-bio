#!/bin/bash
# T1 off vs on in THIS tree on one card, A B B A per size so drift in host load cancels:
# TT_BIO_LEVERS=none (no trunk lever) against the mode default (trimul_t1), warm designs, AICLK sampled inside each.
#   perf/rfd4/tri2/lever_abba.sh <card> <out-dir> [sizes...]     (default sizes: 512 256)
set -u
card=$1 out=$2; shift 2
sizes=${*:-512 256}
tree=$(cd "$(dirname "$0")/../../.." && pwd)
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri2 PYTHONPATH=$tree
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
stamp "start tree=$(git -C "$tree" log --oneline -1 | cut -c1-60)"
leg() {  # leg <name> <levers> <n>
    TT_BIO_LEVERS=$2 timeout 2400 "$py" "$tree/perf/rfd4/e2e/design.py" --spec "$HOME/rfd4-inputs/pdl1-$3.yaml" \
        --weights "$weights" --out "$out/$3/$1" --designs 2 >"$out/$3/$1.log" 2>&1
    stamp "n=$3 $1 rc=$? $(tail -1 "$out/$3/$1/runs.jsonl" 2>/dev/null | cut -c1-400)"
}
for n in $sizes; do
    mkdir -p "$out/$n"
    leg off1 none "$n"; leg t1_1 "" "$n"; leg t1_2 "" "$n"; leg off2 none "$n"
done
stamp end
