#!/bin/bash
# Runs on qb2: refolds the paired cases a featurizer fix changed (refold.json: case -> the commit and why) on one
# main tree, so the tt_bh record is main 90cef828f for every other case (their features are byte-identical) and
# refold.json's tree for these. Folds go to tt-bh-refold first, so neither the running main arm nor the qb1 sync
# (sync_qb1.sh) can mix pre- and post-fix seeds; once both have exited, the pre-fix folds move to tt-bh-prefix,
# the new ones into tt-bh, and their cached grades in work/tt_bh are dropped.
#   bash refold.sh [--book OWNER] MAIN_PID SYNC_PID CHIP [CHIP ...]   log ~/mol1-quality/refold.log, REFOLD-EXIT at the end
# (a finished pid is fine: the waits return at once; --book is passed to run_tt.sh)
set -uo pipefail
BOOK=; [ "${1:-}" = --book ] && { BOOK="--book $2"; shift 2; }
cd ~/mol1-quality; main=${1:?MAIN_PID}; sync=${2:?SYNC_PID}; shift 2; [ $# -gt 0 ] || { echo "no chip"; exit 2; }
P=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality; say() { echo "$(date -u +%FT%TZ) $*"; }
# tt-bio from the engine env (a nohup launch has no login PATH: every fold of the 11-02 run exited 127);
# --use_template aligns hits with kalign (qb2: micromamba env, bioconda kalign3=3.3.5); without it 9fm7 refuses
export PATH=$HOME/tt-bio-dev/env/bin:$HOME/mol1-quality/env/kalign/bin:$PATH
for b in tt-bio kalign; do command -v $b > /dev/null || { say "no $b on PATH"; echo "REFOLD-EXIT rc=1"; exit 1; }; done
SEEDS="42 66 101 2024 8888"; TREE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tree'])" "$P/refold.json")
python3 - "$P/refold.json" > refold-cases.json <<'PY'
import json, sys; print(json.dumps({"paired": [{"id": c} for c in json.load(open(sys.argv[1]))["why"]]}))
PY
# case id -> "fold name (the input file minus .json, as run_tt.sh writes it) and the pdb_id grade.py caches it under"
mapfile -t NAMES < <(python3 - inputs/INDEX.json CASES.json refold-cases.json <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))["cases"]; spec = {c["id"]: c for c in json.load(open(sys.argv[2]))["paired"]}
for c in json.load(open(sys.argv[3]))["paired"]:
    c = spec[c["id"]]; fb = c["source"].startswith("FoldBench/")
    print(idx[c["id"]].removesuffix(".json"), c.get("base", c["id"]) if fb else c["truth"].split(":", 1)[1].split("-")[0] + "-assembly1")
PY
)
if [ ! -d main-$TREE ]; then
  rm -rf main-$TREE.tmp && mkdir main-$TREE.tmp && git -C tt-bio.git archive $TREE tt_bio | tar -x -C main-$TREE.tmp &&
    mv main-$TREE.tmp main-$TREE || { say "tree build failed"; echo "REFOLD-EXIT rc=1"; exit 1; }
fi
say "waiting for main run pid $main (chips $*)"
while kill -0 "$main" 2>/dev/null; do sleep 300; done
if [ -n "$BOOK" ]; then say "waiting for the ${BOOK#--book } window on chip $1"
  until [ "$(python3 ~/spd/feed/book.py active "$1" 2>/dev/null)" = "${BOOK#--book }" ]; do sleep 60; done; fi
say "folding ${#NAMES[@]} cases on $TREE, chips $*"
PYTHONPATH=$PWD/main-$TREE bash "$P/run_tt.sh" $BOOK --cases refold-cases.json inputs tt-bh-refold "$SEEDS" "$@"; rc=$?
# run_tt.sh exits 0 whatever its folds return: a case swaps only when all its seeds are rc=0
bad=0
for x in "${NAMES[@]}"; do n=${x% *}; for s in $SEEDS; do d=tt-bh-refold/.runs/$n/seed_$s
  [ -d $d ] && echo "$TREE" > $d/tree; grep -qx "rc=0" $d/rc 2>/dev/null || { say "not rc=0: $n $s"; bad=1; }; done; done
[ $rc = 0 ] && [ $bad = 0 ] || { say "refold run rc=$rc, failed folds=$bad; nothing swapped, rerun to resume"; echo "REFOLD-EXIT rc=1"; exit 1; }
say "waiting for the qb1 sync pid $sync"
while kill -0 "$sync" 2>/dev/null; do sleep 300; done
mkdir -p tt-bh-prefix/.runs
for x in "${NAMES[@]}"; do
  n=${x% *}; id=${x#* }; r=mol1_results_${n%%.json*}
  for f in .runs/$n $r; do [ -e tt-bh/$f ] && mv tt-bh/$f tt-bh-prefix/$f; [ -e tt-bh-refold/$f ] && mv tt-bh-refold/$f tt-bh/$f; done
  rm -rf work/tt_bh/pp/$n; find work/tt_bh/eval -name "${id}_*" -delete 2>/dev/null  # all refold cases are default-recipe
done
say "swapped ${#NAMES[@]} cases into tt-bh; pre-fix folds in tt-bh-prefix"
echo "REFOLD-EXIT rc=0"
