#!/bin/bash
# p2 exec pass 2, queued ESMC campaign: 4-cell A/B (eager/traced x 0.68/0.75), 2 cycles,
# 50 reps, one pinned card, retry on lease contention, resume-safe. Preference 1,3,0,2.
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
E=artifacts/ttnn075-scout/esmc_ab
mkdir -p $E
rm -f /tmp/p2q_esmc_DONE
DEADLINE=$(( $(date +%s) + 14400 ))

card_free() {
  python3 - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(f"/home/ttuser/.coworker/state/leases/tt-quietbox-card{sys.argv[1]}.json"))
    sys.exit(0 if d.get("released", True) else 1)
except Exception:
    sys.exit(0)
EOF
}

pick_card() {
  if [ -n "${CARD:-}" ] && card_free "$CARD"; then echo "$CARD"; return; fi
  for c in 1 3 0 2; do
    if card_free "$c"; then echo "$c"; return; fi
  done
}

have_json() { [ -s "$1" ] && python3 -c "import json;json.load(open('$1'))" 2>/dev/null; }

CARD=""
cell() {  # ver traceflag tag
  local ver=$1 tf=$2 tag=$3
  have_json $E/$tag.json && { echo "$tag SKIP (exists) $(date -u +%H:%M:%S)" >> $E/queue.log; return 0; }
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    local card
    card=$(pick_card)
    if [ -z "$card" ]; then sleep 60; continue; fi
    CARD=$card
    echo "$tag attempt card=$card $(date -u +%H:%M:%S)" >> $E/queue.log
    SCOUT_CARD=$card timeout -k 15 600 scripts/scout_run_leg.sh $ver \
      scripts/scout_model_timing.py $E/$tag.json 50 $tf >$E/$tag.log 2>&1
    local rc=$?
    echo "$tag rc=$rc card=$card $(date -u +%H:%M:%S)" >> $E/queue.log
    [ $rc -eq 0 ] && return 0
    sleep 45
  done
  return 1
}

for p in 1 2; do
  cell 68 "" esmc_eager68_p$p
  cell 68 "--trace" esmc_trace68_p$p
  cell 75 "" esmc_eager75_p$p
  cell 75 "--trace" esmc_trace75_p$p
done
touch /tmp/p2q_esmc_DONE
