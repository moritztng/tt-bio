#!/bin/bash
# The bfp8 bet's design-level A/B on ONE card in ONE session, every arm a lever set of the same branch
# tree except `main`, which runs main's tt_bio through the branch's design.py (same driver, base code):
#
#   perf/rfd4/bfp8/abab.sh <card> <branch-tree> <main-tree> <out-dir> [sizes...]
#
# Per size: main, branch at its default set, pair_mm, fast (pair_mm,transition_b8), main again (drift).
# Each leg is one process: a compile design, then one warm design per seed. AICLK is sampled inside
# every design (design.py). The default leg must come out bit-identical to main.
set -u
card=$1 tree=$2 mtree=$3 out=$4; shift 4
sizes=${*:-128 512}
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-perf-bfp8}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"; }
stamp "start branch=$(git -C "$tree" log --oneline -1) main=$(git -C "$mtree" log --oneline -1)"

leg() {  # leg <size> <name> <code-tree> <levers or -> <seeds...>
    local n=$1 name=$2 code=$3 lv=$4; shift 4
    local env=(); [ "$lv" != - ] && env=(TT_BIO_LEVERS="$lv")
    (cd "$code" && env "${env[@]}" PYTHONPATH="$code" timeout 5400 "$py" "$tree/perf/rfd4/e2e/design.py" \
        --spec "$HOME/rfd4-inputs/pdl1-$n.yaml" --weights "$weights" --out "$out/$n/$name" --rng upstream \
        --seeds "$@") >"$out/$n/$name.log" 2>&1
    stamp "$n $name rc=$? warm=$(grep -c '"kind": "warm"' "$out/$n/$name/runs.jsonl" 2>/dev/null)"
}
for n in $sizes; do
    mkdir -p "$out/$n"
    if [ "$n" -le 128 ]; then seeds=$(seq 0 9); else seeds=$(seq 0 7); fi
    leg "$n" main "$mtree" - $seeds
    leg "$n" default "$tree" - $seeds
    leg "$n" pair_mm "$tree" pair_mm $seeds
    leg "$n" fast "$tree" pair_mm,transition_b8 $seeds
    leg "$n" main2 "$mtree" - 0 1
done
stamp end
