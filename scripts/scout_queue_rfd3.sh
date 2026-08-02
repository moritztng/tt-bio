#!/bin/bash
# p2 exec pass 2, queued RFD3 campaign: waits for a free card, pins it, runs the owed
# leg-per-process A/B with retry. Resume-safe: skips legs whose JSON already exists.
# Global deadline 4h. Card preference 0,2,1,3 (ESMC campaign prefers 1,3).
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
export TT_BIO_TRACE_REGION_SIZE=$((1<<30))
R=artifacts/ttnn075-scout/rfd3_ab
mkdir -p $R
rm -f /tmp/p2q_rfd3_DONE
DEADLINE=$(( $(date +%s) + 14400 ))

card_free() {  # $1 = card idx; lease json missing/released => free
  python3 - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(f"/home/ttuser/.coworker/state/leases/tt-quietbox-card{sys.argv[1]}.json"))
    sys.exit(0 if d.get("released", True) else 1)
except Exception:
    sys.exit(0)
EOF
}

pick_card() {  # sticky: reuse CARD if still free, else first free by preference
  if [ -n "${CARD:-}" ] && card_free "$CARD"; then echo "$CARD"; return; fi
  for c in 0 2 1 3; do
    if card_free "$c"; then echo "$c"; return; fi
  done
}

have_json() { [ -s "$1" ] && python3 -c "import json;json.load(open('$1'))" 2>/dev/null; }

CARD=""
leg() {  # ver legname tag
  local ver=$1 ln=$2 tag=$3
  have_json $R/$tag.json && { echo "$tag SKIP (exists) $(date -u +%H:%M:%S)" >> $R/queue.log; return 0; }
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    local card
    card=$(pick_card)
    if [ -z "$card" ]; then sleep 60; continue; fi
    CARD=$card
    echo "$tag attempt card=$card $(date -u +%H:%M:%S)" >> $R/queue.log
    SCOUT_CARD=$card timeout -k 15 900 scripts/scout_run_leg.sh $ver \
      scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 \
      --legs $ln --alternations 2 --json_out $R/$tag.json >$R/$tag.log 2>&1
    local rc=$?
    echo "$tag rc=$rc card=$card $(date -u +%H:%M:%S)" >> $R/queue.log
    [ $rc -eq 0 ] && return 0
    sleep 45
  done
  return 1
}

leg 68 eager rfd3_eager68_p1
leg 75 eager rfd3_eager75_p1
leg 68 eager rfd3_eager68_p2
leg 75 eager rfd3_eager75_p2
leg 68 both rfd3_both68_fresh
leg 75 both rfd3_both75_fresh
touch /tmp/p2q_rfd3_DONE
