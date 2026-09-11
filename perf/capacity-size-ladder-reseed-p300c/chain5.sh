#!/bin/sh
# protenix-v2 is the one ladder model chain4 cannot record: it crashes ~47 s in, both times it
# has been run, and both times it ran directly after another model in the shared workdir. It has
# to go last because it needs the baseline file to itself -- two recorders writing
# docs/size_ladder_baseline.json at once is a write race on the deliverable.
#
# Two attempts. First the plain CLI standalone, which tests the "only fails when it follows
# another ladder model" hypothesis at zero extra cost. If that crashes the same way, the second
# runs the same recorder code with SIZE_LADDER_WORKDIR pointed somewhere private, which is the
# configuration that walked all six rungs clean on card 2 this morning.
WT=/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c
OUT=$WT/perf/capacity-size-ladder-reseed-p300c
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PRIOR=9071
cd "$WT" || exit 1

log() { printf "%s %s\n" "$(date -u +%FT%TZ)" "$*" >> "$OUT/chain5.log"; return 0; }

have_ladder() {
  $PY -c "import json,sys
try: d=json.load(open(\"docs/size_ladder_baseline.json\"))
except Exception: sys.exit(1)
e=(d.get(\"cards\") or {}).get(\"p300c\",{}).get(\"models\",{}).get(\"protenix-v2\")
if not e or (e.get(\"recorded\") or \"\") < \"2026-09-10\": sys.exit(1)
want={256,512,640,768,896,1024}
got={int(r) for r in (e.get(\"runtime_s\") or {})} | {int(r) for r in (e.get(\"refused\") or {})}
sys.exit(0 if want<=got else 1)"
}

log "chain5 start (pid $$), waiting on chain4 pid $PRIOR"
while kill -0 "$PRIOR" 2>/dev/null; do sleep 60; done
log "chain4 exited"

if have_ladder; then log "protenix-v2 already recorded, nothing to do"; exit 0; fi

log "attempt 1: plain CLI, standalone"
TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
PYTHONPATH=$WT $PY scripts/release_gate.py --model size-ladder \
  --size-ladder-record --size-ladder-models protenix-v2 \
  > "$OUT/ladder_protenix-v2_c5a.log" 2>&1
log "attempt 1 rc=$?"
if have_ladder; then log "protenix-v2 recorded by attempt 1 (standalone is enough)"; exit 0; fi

log "attempt 2: private workdir"
TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
PYTHONPATH=$WT $PY "$OUT/pv2_record.py" > "$OUT/ladder_protenix-v2_c5b.log" 2>&1
log "attempt 2 rc=$?"
if have_ladder; then log "protenix-v2 recorded by attempt 2 (private workdir)"
else log "protenix-v2 STILL NOT recorded, see both logs"; fi
log "chain5 complete"
