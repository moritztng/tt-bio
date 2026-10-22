#!/usr/bin/env bash
# Card action 0 for rfd4-beam: does the OpenFold3 serving path still fold the same thing
# after the host/device halves were lifted out of `_predict_openfold3_one`?
#
# Folds the same fixture, same seed, same flags, on the same card, in two trees:
#   /tmp/of3-premove  (5dc685d76 = 3d9d63cdf~1, before the move)
#   this worktree     (after the move, with the oracle backend on top)
# then diffs the metrics rows. Samples AICLK throughout, because a number without a clock
# is not a measurement -- here it only tells us whether the box was clamped, the verdict
# itself is a correctness diff and does not depend on the clock.
set -u

WT=/home/ttuser/.coworker/wt/rfd4-beam
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=3
OUT=$WT/perf/rfd4/results
mkdir -p "$OUT"

export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:rfd4-beam

# --- AICLK sampler, every 20 s, for as long as this script lives -------------------
CLK=$OUT/of3_verify_clock.jsonl
: > "$CLK"
(
  while :; do
    ts=$(date -u +%FT%TZ)
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | "$PY" -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
rows=[]
for i,b in enumerate(d.get('device_info',[])):
    t=b.get('telemetry',{})
    rows.append({'idx':i,'bdf':b.get('board_info',{}).get('bus_id'),
                 'type':b.get('board_info',{}).get('board_type'),
                 'aiclk':t.get('aiclk'),'power':t.get('power'),'asic_temp':t.get('asic_temperature')})
print(json.dumps({'ts':'$ts','boards':rows}))
" >> "$CLK"
    sleep 20
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

cd "$WT" || exit 1

echo "=== [$(date -u +%FT%TZ)] premove fold (5dc769..., tree /tmp/of3-premove) on card $CARD"
timeout 3600 "$PY" perf/rfd4/verify_of3_refactor.py fold --card "$CARD" \
    --tree /tmp/of3-premove --out "$OUT/of3_before.json"
RC_B=$?
echo "=== premove rc=$RC_B"

echo "=== [$(date -u +%FT%TZ)] head fold (this worktree) on card $CARD"
timeout 3600 "$PY" perf/rfd4/verify_of3_refactor.py fold --card "$CARD" \
    --out "$OUT/of3_after.json"
RC_A=$?
echo "=== head rc=$RC_A"

if [ $RC_B -eq 0 ] && [ $RC_A -eq 0 ]; then
  echo "=== [$(date -u +%FT%TZ)] compare"
  "$PY" perf/rfd4/verify_of3_refactor.py compare "$OUT/of3_before.json" "$OUT/of3_after.json"
  echo "=== compare rc=$?"
else
  echo "=== compare SKIPPED, a fold failed"
fi

echo "=== AICLK seen during the run:"
"$PY" - "$CLK" <<'PYEOF'
import json,sys
vals={}
for line in open(sys.argv[1]):
    try: d=json.loads(line)
    except Exception: continue
    for b in d['boards']:
        if b.get('aiclk') is not None:
            vals.setdefault(b['idx'],[]).append(float(b['aiclk']))
for i,v in sorted(vals.items()):
    print(f"  board {i}: n={len(v)} min={min(v)} max={max(v)} last={v[-1]}")
PYEOF
echo "=== [$(date -u +%FT%TZ)] DONE_SCRIPT"
