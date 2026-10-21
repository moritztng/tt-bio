#!/bin/bash
# A/B one RFD4 perf branch against its base on ONE card in ONE session: device suites on the branch,
# then warm designs base, branch, base (the second base catches drift on a shared board), then the
# census on the branch. AICLK is sampled inside every design (design.py) and census leg.
#
#   perf/rfd4/e2e/ab.sh <card> <base-commit> <branch-tree> <base-tree> <out-dir>
#
# Both trees must already be checked out (`git worktree add`); this script does not move either.
set -u
card=$1 base=$2 tree=$3 btree=$4 out=$5
py=${RFD4_PY:-$HOME/tt-bio-dev/env/bin/python}
spec=${RFD4_SPEC:-$HOME/rfd4-inputs/pdl1-128.yaml}
weights=${RFD4_WEIGHTS:-$HOME/rfd4-bundle}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rfd4-ab}
stamp() { echo "$1 $(date -u +%FT%TZ)"; }

stamp "start branch=$(git -C "$tree" log --oneline -1) base=$(git -C "$btree" log --oneline -1)"
[ "$(git -C "$btree" rev-parse HEAD)" = "$(git -C "$btree" rev-parse "$base")" ] || { echo "base tree not at $base"; exit 2; }

(cd "$tree" && PYTHONPATH=$PWD timeout 1800 "$py" -m pytest -q -m device \
    tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py) >"$out/suites.log" 2>&1
stamp "suites rc=$? $(tail -1 "$out/suites.log")"

leg() {  # leg <name> <tree>
    (cd "$2" && PYTHONPATH=$PWD timeout 1200 "$py" perf/rfd4/e2e/design.py \
        --spec "$spec" --weights "$weights" --out "$out/$1" --designs 2) >"$out/$1.log" 2>&1
    stamp "design $1 rc=$? $(tail -1 "$out/$1/runs.jsonl" 2>/dev/null)"
}
leg base1 "$btree"
leg branch "$tree"
leg base2 "$btree"

(cd "$tree" && PYTHONPATH=$PWD timeout 3600 "$py" perf/rfd4/e2e/census.py \
    --spec "$spec" --weights "$weights" --out "$out/census.json") >"$out/census.log" 2>&1
stamp "census rc=$?"
stamp end
