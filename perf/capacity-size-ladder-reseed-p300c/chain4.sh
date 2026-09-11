#!/bin/sh
# Resume after qb2 hung at 06:57Z mid-campaign (silent host hang, box came back 09:54Z).
# chain.sh got 3 of 9 ladder models in; chain3.sh never ran a thing. This picks up the rest.
#
# Ordered cheapest first, deliberately: the box hangs about once a day, so the ordering that
# banks the most recorded cells per hour of uptime is the one that survives it.
#
# Completion is artifact-keyed, never rc: both gates exit nonzero when a cell legitimately
# FAILs, so an rc-keyed marker re-runs a finished model forever. A ladder model is done when
# its p300c entry holds every rung the ladder walks AND was recorded on/after 2026-09-10, which
# is this campaign. Keying on the commit (chain.sh/chain3.sh did) breaks the moment anything
# commits mid-campaign, and the reboot made committing the measured data the right call.
WT=/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c
OUT=$WT/perf/capacity-size-ladder-reseed-p300c
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CAMPAIGN=2026-09-10
cd "$WT" || exit 1

log() { printf "%s %s\n" "$(date -u +%FT%TZ)" "$*" >> "$OUT/chain4.log"; return 0; }

have_cell() {
  $PY -c "import json,sys
try: d=json.load(open(sys.argv[2]))
except Exception: sys.exit(1)
sys.exit(0 if any(r.get(\"model\")==sys.argv[1] for r in d.get(\"results\") or []) else 1)" "$1" "$2"
}

have_ladder() {
  $PY -c "import json,sys
m=sys.argv[1]; campaign=sys.argv[2]
try: d=json.load(open(\"docs/size_ladder_baseline.json\"))
except Exception: sys.exit(1)
e=(d.get(\"cards\") or {}).get(\"p300c\",{}).get(\"models\",{}).get(m)
if not e or (e.get(\"recorded\") or \"\") < campaign: sys.exit(1)
want={256,512,640,768,896,1024} | ({1088} if m==\"rf3\" else set())
got={int(r) for r in (e.get(\"runtime_s\") or {})} | {int(r) for r in (e.get(\"refused\") or {})}
sys.exit(0 if want<=got else 1)" "$1" "$CAMPAIGN"
}

ladder() {
  m=$1; shift
  if have_ladder "$m"; then log "$m ladder already recorded, skipping"; return 0; fi
  log "$m ladder start"
  TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
  TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
  PYTHONPATH=$WT $PY scripts/release_gate.py --model size-ladder \
    --size-ladder-record --size-ladder-models "$m" "$@" \
    > "$OUT/ladder_${m}_c4.log" 2>&1
  log "$m ladder rc=$?"
  if have_ladder "$m"; then log "$m ladder recorded"
  else log "$m ladder NOT recorded: $(grep -o "NOT RECORDED.*" "$OUT/ladder_${m}_c4.log" | head -1)"; fi
}

capcell() {
  m=$1
  if have_cell "$m" "$OUT/capgate_recard_$m.json"; then
    log "$m capacity already re-measured, skipping"; return 0; fi
  log "$m capacity re-measure start"
  TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
  PYTHONPATH=$WT $PY scripts/capacity_gate.py --workers tt-quietbox2:3 --models "$m" --record \
    --work-dir "$OUT/work" --report "$OUT/capgate_recard_$m.json" > "$OUT/recard_$m.log" 2>&1
  log "$m capacity rc=$?"
  if have_cell "$m" "$OUT/capgate_recard_$m.json"; then
    log "$m: $($PY -c "import json,sys
d=json.load(open(sys.argv[1]))
c=[r for r in d[\"results\"] if r[\"model\"]==sys.argv[2]][0]
print(c.get(\"verdict\"), \"mechanism\", c.get(\"mechanism\"), \"ceiling\", c.get(\"ceiling_tokens\"),
      \"|\", (c.get(\"first_error\") or \"\")[:140])" "$OUT/capgate_recard_$m.json" "$m")"
  else
    log "$m capacity NOT re-measured"
  fi
}

log "chain4 start (pid $$), HEAD $(git rev-parse --short HEAD)"

ladder nesso1

# protenix-v2 crashed at rung 256 rep0 inside release_gate.py error path: the fold exited
# nonzero and its own stdout log was gone by the time line 2254 read it, so the fold error was
# destroyed with it. --keep leaves the workdir; the watcher samples it every second, which is
# what a static read of the gate could not settle.
if have_ladder protenix-v2; then
  log "protenix-v2 ladder already recorded, skipping"
else
  ( while :; do
      printf "%s %s\n" "$(date -u +%FT%T.%NZ)" \
        "$(ls "$WT/perf/sizegate/work" 2>&1 | tr "\n" " ")" >> "$OUT/pv2_workdir_watch.log"
      sleep 1
    done ) &
  WATCH=$!
  ladder protenix-v2 --keep
  kill "$WATCH" 2>/dev/null
fi

capcell protenix-v1
ladder openfold3
ladder openbind
ladder rf3
# opendde failed at the 1024 warm-up with a 1800 s census-fold timeout at 06:57:43Z, which is the
# minute the host hung. Its 768 rung took 184 s, so 1024 should land near 320 s, not past 1800 s.
# Re-run rather than record a timeout as a wall.
ladder opendde
capcell opendde
capcell opendde-abag

log "chain4 complete"
