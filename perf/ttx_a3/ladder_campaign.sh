#!/usr/bin/env bash
# Splice SDPA_FUSED_LARGE_S into one ladder model's p300c record and re-check that model.
#
# A new census counter shows up in every ladder model's census, so every model's record needs a
# row for it before check mode can pass -- that is what `--size-ladder-record-lever` is for. Three
# steps per model, each resumable through the shared progress file: splice (one fold per rung),
# author the one judgement the dark rungs need and let release_gate carry it up, then check.
#
# CARD=<n> picks the card; the lease grant widens to 0,<n> off card 0.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate6"; PROG="$OUT/progress"; mkdir -p "$OUT"; touch "$PROG"
CARD="${CARD:-0}"
export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
if [ "$CARD" = 0 ]; then export TT_BIO_LEASE_CARDS=0; else export TT_BIO_LEASE_CARDS="0,$CARD"; fi
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2
P=/home/ttuser/tt-bio-dev/env/bin/python3
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

run() {  # $1 = arm name, rest = argv
  local name="$1"; shift
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  log "$name START card=$CARD loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  TT_VISIBLE_DEVICES=$CARD timeout -k 30 "${ARM_TIMEOUT:-3000}" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  log "$name rc=$rc loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  echo "ARM $name rc=$rc"
  # `timeout` signals only its direct child, so a fold that forked a device worker leaves the
  # worker holding the card. Reap by EXPLICIT pid, never a pattern: a sibling arm is on another
  # card and a pkill would take it too.
  for p in $(lsof -t /dev/tenstorrent/* 2>/dev/null); do
    grep -qa "TT_VISIBLE_DEVICES=$CARD" "/proc/$p/environ" 2>/dev/null || continue
    echo "REAP leaked $p after $name"; kill -TERM "$p" 2>/dev/null
  done
  return $rc
}

for m in "$@"; do
  frag="$WT/docs/size_ladder_baseline.d/$m.json"
  [ -f "$frag" ] || { echo "no fragment for $m"; continue; }
  run "splice-$m" $P scripts/release_gate.py --model size-ladder \
      --size-ladder-record-lever SDPA_FUSED_LARGE_S --size-ladder-models "$m" || continue
  $P perf/ttx_a3/ladder_reason.py "$frag" p300c >> "$OUT/reasons.log" 2>&1
  $P scripts/release_gate.py --model size-ladder --size-ladder-fill-reasons \
      --size-ladder-models "$m" >> "$OUT/reasons.log" 2>&1
  ARM_TIMEOUT=3600 run "ladder-$m" $P scripts/release_gate.py --model size-ladder \
      --size-ladder-models "$m"
done
log "LADDER_CAMPAIGN_DONE card=$CARD models=$*"
