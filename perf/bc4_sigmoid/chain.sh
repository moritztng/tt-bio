#!/bin/bash
# bc4-bet-sigmoid, the whole device chain on one leased qb2 chip, detached:
#   CARD=<chip> setsid nohup bash perf/bc4_sigmoid/chain.sh > /dev/null 2>&1 &
# 1. f64_check (float64 grade + per-op timing); a hard stop (output outside [0,1], NaN, sign error) ends it.
# 2. in-process ABBA, TT_BIO_SIGMOID_FAST off/on per rep (bc2_evo_devprof --alternate), AICLK at 1 Hz.
# 3. design grade: bench.py bindcraft2 seeds 101-103, warm 9, 80 aa, flag on (graded on pc by grade.sh).
set -u
CARD=${CARD:?}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
SHA=$(git rev-parse --short HEAD); D=$WT/_sig/$SHA; mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-sigmoid TT_BIO_LEASE_TIMEOUT=7200
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/chain.log"; }
log "start pid $$ chip $CARD sha $(git rev-parse HEAD) dirty $(git status --porcelain -uno | wc -l)"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & CLKP=$!
trap 'kill $CLKP 2>/dev/null' EXIT

timeout 1500 "$PY" perf/bc4_sigmoid/f64_check.py --out "$D/f64.jsonl" > "$D/f64.log" 2>&1
log "f64 rc=$?"
"$PY" - "$D/f64.jsonl" > "$D/hardstop.txt" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [r["arm"] for r in rows if "arm" in r and "lut" in r["arm"]
       and (r.get("outside_01", 0) or r.get("nan", 0) or r.get("sign_err", 0) or "error" in r)]
lut = [r for r in rows if "arm" in r and "lut" in r["arm"] and "error" not in r]
print("HARDSTOP" if bad or not lut else "OK", bad, len(lut))
PYEOF
log "hardstop: $(cat "$D/hardstop.txt")"
grep -q '^OK' "$D/hardstop.txt" || { log "stop: LUT failed the float64 hard stop"; touch "$D/done"; exit 0; }

timeout -s INT -k 120 9000 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/alt" \
    --warm 4 --reps "${REPS:-24}" --alternate tt_bio.gate_sigmoid.SIGMOID_FAST > "$D/alt.log" 2>&1
log "alt rc=$?"

for s in 101 102 103; do
  mkdir -p "$D/q"; log "s$s start load $(cut -d' ' -f1 /proc/loadavg)"
  TT_BIO_SIGMOID_FAST=1 timeout 21600 "$PY" perf/spd/bench.py --out "$D/q/tt_s$s" --chip "$CARD" --model bindcraft2 \
      --bc2 "$HOME/bcx_e2e/bc2" --seed $s --warm 9 --inputs 80 > "$D/q/tt_s$s.log" 2>&1
  log "s$s rc=$?"
done
touch "$D/done"; log "done"
