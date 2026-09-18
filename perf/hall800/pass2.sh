#!/usr/bin/env bash
# Pass-2 device session on the free sibling card: census first (one fold, converts the
# dark-gate reading from source to measurement), then reps for a spread on the fold time.
# Census goes first deliberately -- 800 aa can host-spin-wedge, and if one fold is going to
# be lost it should be a rep and not the census.
set -uo pipefail
cd /home/ttuser/.coworker/wt/hall-capacity-800aa
CARD=${1:?card}
echo "== pass2 on card $CARD, start $(date -Is)"
bash perf/hall800/census.sh h800 perf/hall800/hall800.yaml "$CARD"
echo "== census done rc=$? $(date -Is)"
bash perf/hall800/run.sh solo3c"$CARD" 3 --cards "$CARD"
echo "== reps done rc=$? $(date -Is)"
