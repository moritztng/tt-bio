#!/bin/bash
# The pair built on the card against the host pair, on ONE card in ONE session: device suites on the
# branch, then host_split legs ABAB at 128 and at 512 (AICLK sampled inside every design, host load
# recorded), then one other-seed base design per size for the seed floor and the Angstrom grade.
#
#   perf/rfd4/e2e/pairdev_ab.sh <card> <branch-tree> <base-tree> <out-dir>
#
# Holds ~/spd_qb2_chip<card>.lock for the whole run; every command has a timeout.
set -u
card=$1 tree=$2 btree=$3 out=$4
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
inputs=${RFD4_INPUTS:-$HOME/rfd4-inputs}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-perf-pairdev}
exec 9>"$HOME/spd_qb2_chip$card.lock"
flock 9
stamp() { echo "$1 $(date -u +%FT%TZ)" | tee -a "$out/campaign.log"; }
stamp "start card=$card branch=$(git -C "$tree" log --oneline -1) base=$(git -C "$btree" log --oneline -1) load=$(cut -d' ' -f1 /proc/loadavg)"

(cd "$tree" && PYTHONPATH=$PWD timeout 2400 "$py" -m pytest -q -s -m device \
    tests/test_rfd4_pair_device.py tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py) >"$out/suites.log" 2>&1
stamp "suites rc=$? $(tail -1 "$out/suites.log")"

leg() {  # leg <name> <tree> <size> <designs> <seed>
    (cd "$2" && PYTHONPATH=$PWD timeout 3000 "$py" perf/rfd4/e2e/host_split.py \
        --spec "$inputs/pdl1-$3.yaml" --weights "$weights" --out "$out/$1" --designs "$4" --seed "$5" --tag "$1") \
        >"$out/$1.log" 2>&1
    stamp "leg $1 rc=$? $(tail -1 "$out/$1/split.jsonl" 2>/dev/null | cut -c1-400)"
}
for r in 1 2; do
    leg base128_$r "$btree" 128 3 0
    leg pair128_$r "$tree" 128 3 0
done
for r in 1 2; do
    leg base512_$r "$btree" 512 2 0
    leg pair512_$r "$tree" 512 2 0
done
leg floor128 "$btree" 128 1 1
leg floor512 "$btree" 512 1 1

for n in 128 512; do
    ours=$(ls "$out/pair${n}_1"/*warm*.cif | head -1)
    base=$(ls "$out/base${n}_1"/*warm*.cif | head -1)
    floor=$(ls "$out/floor$n"/*.cif | head -1)
    (cd "$tree" && timeout 600 "$py" perf/rfd4/e2e/grade.py --ours "$ours" --upstream "$base" --floor "$floor" \
        --out "$out/grade$n.json") >"$out/grade$n.log" 2>&1
    stamp "grade $n rc=$? $(tail -1 "$out/grade$n.log" | cut -c1-300)"
done
stamp "done"
