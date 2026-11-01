#!/bin/bash
# Runs on qb2: refolds the cases mol1-plddt's terminal-cap fix changes (ACE/NME bonded in token_bonds, the pre-NME OXT
# kept; main dc1873acb) on the tt_bh arm's engine, main 90cef828f, with only tt_bio/protenix_data.py taken from
# 89f21787c. Every other case's features are byte-identical, so its folds stand.
#   bash cap_refold.sh MAIN_PID CHIP [CHIP ...]     log ~/mol1-quality/cap.log, CAP-EXIT at the end
# It claims those case+seeds at once (run_tt.sh skips a claim whose pid lives), so the running main fold leaves them
# alone; once MAIN_PID exits it moves the pre-fix folds to tt-bh-precap, drops their cached grades in work/tt_bh and
# folds them into tt-bh on CHIPs. Each new seed dir gets a `tree` file naming the engine.
set -uo pipefail
cd ~/mol1-quality; main=${1:?MAIN_PID}; shift; [ $# -gt 0 ] || { echo "no chip"; exit 2; }
CASES="7y37-assembly1 8q9n-assembly1"; SEEDS="42 66 101 2024 8888"
TREE="90cef828f + tt_bio/protenix_data.py of 89f21787c"; say() { echo "$(date -u +%FT%TZ) $*"; }
for c in $CASES; do for s in $SEEDS; do
  d=tt-bh/.runs/$c/seed_$s; mkdir -p "$d"
  grep -qs rc=0 "$d/rc" || echo $$ > "$d/claim"
done; done
say "claimed; waiting for main run pid $main"
while kill -0 "$main" 2>/dev/null; do sleep 300; done
if [ ! -d main-cap ]; then
  cp -a main main-cap.tmp && rm -rf main-cap.tmp/tt_bio/__pycache__/protenix_data.* &&
    git -C tt-bio.git show 89f21787c:tt_bio/protenix_data.py > main-cap.tmp/tt_bio/protenix_data.py &&
    echo "$TREE" > main-cap.tmp/TREE_NOTE && mv main-cap.tmp main-cap || { say "tree build failed"; echo "CAP-EXIT rc=1"; exit 1; }
fi
mkdir -p tt-bh-precap/.runs
for c in $CASES; do
  [ -e tt-bh/.runs/$c ] && mv tt-bh/.runs/$c tt-bh-precap/.runs/
  [ -e tt-bh/mol1_results_$c ] && mv tt-bh/mol1_results_$c tt-bh-precap/
  rm -rf work/tt_bh/pp/$c; find work/tt_bh -name "${c}_*" -delete 2>/dev/null
done
python3 -c "import json,sys; print(json.dumps({'paired': [{'id': c} for c in sys.argv[1:]]}))" $CASES > cap-cases.json
say "folding $CASES on $TREE, chips $*"
PYTHONPATH=$PWD/main-cap bash code/run_tt.sh --cases cap-cases.json inputs tt-bh "$SEEDS" "$@"; rc=$?
for c in $CASES; do for s in $SEEDS; do echo "$TREE" > tt-bh/.runs/$c/seed_$s/tree; done; done
echo "CAP-EXIT rc=$rc"
