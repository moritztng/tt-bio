#!/bin/bash
# Bet 6's whole device phase on ONE card, detached: the op bench, the dit_pair_bias bit-compare against
# the base tree, the device suites, interleaved base/branch designs (A B A B, seed 0, upstream RNG so the
# CIF pairs with upstream's), then the census on both trees. AICLK is sampled inside every leg.
#
#   perf/rfd4/e2e/attn_ab.sh <card> <base-commit> <out-dir>
set -u
card=$1 base=$2 out=$3
wt=$(cd "$(dirname "$0")/../../.." && pwd)
btree=$wt/artifacts/base
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
spec=${RFD4_SPEC:-$HOME/rfd4-inputs/pdl1-128.yaml}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-perf-attn}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"; }
[ -d "$btree" ] || git -C "$wt" worktree add --detach "$btree" "$base" >/dev/null 2>&1
[ "$(git -C "$btree" rev-parse HEAD)" = "$(git -C "$wt" rev-parse "$base")" ] || { echo "base tree not at $base"; exit 2; }
stamp "start branch=$(git -C "$wt" log --oneline -1) base=$(git -C "$btree" log --oneline -1)"

(cd "$wt" && PYTHONPATH=$wt timeout 900 "$py" perf/rfd4/e2e/attn_bench.py --weights "$weights" --out "$out/bench.json") \
    >"$out/bench.log" 2>&1
stamp "bench rc=$?"
for side in branch base; do
    tree=$wt; [ $side = base ] && tree=$btree
    (cd "$tree" && PYTHONPATH=$tree timeout 600 "$py" "$wt/perf/rfd4/e2e/dpb_regress.py" --out "$out/dpb_$side.pt") \
        >"$out/dpb_$side.log" 2>&1
    stamp "dpb $side rc=$?"
done
(cd "$wt" && PYTHONPATH=$wt timeout 2400 "$py" -m pytest -q -s -m device \
    tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py) >"$out/suites.log" 2>&1
stamp "suites rc=$? $(tail -1 "$out/suites.log")"

for round in 1 2 3; do for side in base branch; do
    tree=$wt; [ $side = base ] && tree=$btree
    stamp "design $side r$round"
    (cd "$tree" && PYTHONPATH=$tree timeout 1800 "$py" "$wt/perf/rfd4/e2e/design.py" --spec "$spec" \
        --weights "$weights" --out "$out/abab_${side}_r$round" --seed 0 --designs 2 --rng upstream) \
        >"$out/abab_${side}_r$round.log" 2>&1
    stamp "design $side r$round rc=$? $(grep '"kind": "warm"' "$out/abab_${side}_r$round.log" | tail -1)"
done; done

for side in branch base; do
    tree=$wt; [ $side = base ] && tree=$btree
    (cd "$tree" && PYTHONPATH=$tree timeout 3600 "$py" "$wt/perf/rfd4/e2e/census.py" --spec "$spec" \
        --weights "$weights" --out "$out/census_$side.json" --work "$out/census_work_$side") >"$out/census_$side.log" 2>&1
    stamp "census $side rc=$?"
done
stamp end
