#!/bin/bash
# Paired multi-seed grade of the card-built pair against the host pair, plus a 256 ABAB for time.
# Seed-matched designs on both trees, then grade_seeds (CA RMSD per seed beside the seed floor) and the
# oracle-free quality grade, as rfd4-perf-tri2 graded its 512 stack.
#
#   perf/rfd4/e2e/pairdev_seeds.sh <card> <branch-tree> <base-tree> <out-dir>
#
# Uses the seed-capable design.py / grade_seeds.py / quality.py of the harness trees named below
# (RFD4_HARNESS, RFD4_QHARNESS) against each tree's own engine (PYTHONPATH). Holds the chip lock.
set -u
card=$1 tree=$2 btree=$3 out=$4
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
h=${RFD4_HARNESS:-$HOME/rfd4-tri2-runs/harness}/perf/rfd4
q=${RFD4_QHARNESS:-$HOME/rfd4-tri2-runs/qharness}/perf/rfd4
g=${RFD4_GRADE:-$HOME/.coworker/wt/rfd4-perf-bfp8/perf/rfd4/bfp8/grade_seeds.py}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-perf-pairdev}
exec 9>"$HOME/spd_qb2_chip$card.lock"
flock 9
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$out/campaign.log"; }
stamp "start branch=$(git -C "$tree" log --oneline -1 | cut -c1-60) base=$(git -C "$btree" log --oneline -1 | cut -c1-60)"

(cd "$tree" && PYTHONPATH=$PWD timeout 900 "$py" -m pytest -q -s -m device tests/test_rfd4_pair_device.py) \
    >"$out/pair_device.log" 2>&1
stamp "pair_device rc=$? $(tail -1 "$out/pair_device.log")"

for r in 1 2; do  # 256 time, interleaved
    for arm in base pair; do
        t=$([ $arm = base ] && echo "$btree" || echo "$tree")
        (cd "$t" && PYTHONPATH=$PWD timeout 1800 "$py" perf/rfd4/e2e/host_split.py --spec ~/rfd4-inputs/pdl1-256.yaml \
            --weights ~/rfd4-bundle --out "$out/${arm}256_$r" --designs 2 --seed 0 --tag "${arm}256_$r") \
            >"$out/${arm}256_$r.log" 2>&1
        stamp "leg ${arm}256_$r rc=$? $(tail -1 "$out/${arm}256_$r/split.jsonl" 2>/dev/null | cut -c1-300)"
    done
done

for n in 512 256; do
    seeds=$([ $n = 256 ] && echo "0 1 2 3 4 5 6 7" || echo "0 1 2 3 4 5")
    for arm in base pair; do
        t=$([ $arm = base ] && echo "$btree" || echo "$tree")
        (cd "$t" && PYTHONPATH=$PWD timeout 7200 "$py" "$h/e2e/design.py" --spec ~/rfd4-inputs/pdl1-$n.yaml \
            --weights ~/rfd4-bundle --out "$out/seeds/$n/$arm" --seeds $seeds) >"$out/seeds-$n-$arm.log" 2>&1
        stamp "seeds n=$n arm=$arm rc=$?"
    done
    "$py" "$g" --base "$out/seeds/$n/base" --arm "$out/seeds/$n/pair" --out "$out/grade-$n.json" \
        >"$out/grade-$n.log" 2>&1
    stamp "grade n=$n rc=$? $(tail -1 "$out/grade-$n.log" | cut -c1-300)"
    (cd "$tree" && timeout 900 "$py" "$q/quality.py" grade --feature binder --tt "$out/seeds/$n/pair" \
        --ref "$out/seeds/$n/base" --label pairdev --n-min 6 --allow-no-oracle --allow-no-floor --out "$out/quality-$n.json") >"$out/quality-$n.log" 2>&1
    stamp "quality n=$n rc=$? $(tail -1 "$out/quality-$n.log" | cut -c1-300)"
done
stamp done
