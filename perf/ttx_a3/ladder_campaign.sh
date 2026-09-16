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

recorded_rc() {  # last recorded rc for an arm name, empty if it never ran
  grep " $1 rc=" "$PROG" | tail -1 | sed -n 's/.* rc=\([0-9]*\).*/\1/p'
}

# THREE ATTEMPTS PER MODEL, because one wedge is not a verdict and a single-shot arm cannot
# survive this box. boltz-2 wedges at `trunk 0/4` roughly one rung in six -- 08:16 (896), 08:29
# (640), 08:46 (640), 10:09 (1024 warm-up), 12:41 (768) -- and it is NOT load, whatever the first
# reading of it said: the 12:41 wedge happened at loadavg 1.1 with this gate as the only lane on
# the box, and the same tree walked all six rungs clean at 07:52-07:59. A ladder arm is six rungs,
# so at that rate a single attempt fails more often than it passes and every model would record a
# red that says nothing about the lever.
#
# Attempts are SEPARATE arm names, so the progress file keeps each one and a resume re-reads them
# instead of re-running them. `run` returns 0 when it skips an already-recorded arm, so the
# verdict has to come from the recorded rc, never from run's exit status.
ladder_with_retries() {
  local m="$1" att name rc
  for att in 1 2 3; do
    name="ladder-$m"; [ "$att" -gt 1 ] && name="ladder-$m-att$att"
    rc=$(recorded_rc "$name")
    if [ -z "$rc" ]; then
      ARM_TIMEOUT=3600 run "$name" $P scripts/release_gate.py --model size-ladder \
          --size-ladder-models "$m"
      rc=$(recorded_rc "$name")
    fi
    [ "$rc" = 0 ] && { log "ladder-$m PASS on attempt $att"; return 0; }
  done
  log "ladder-$m RED after 3 attempts"
  return 1
}

for m in "$@"; do
  frag="$WT/docs/size_ladder_baseline.d/$m.json"
  [ -f "$frag" ] || { echo "no fragment for $m"; continue; }
  run "splice-$m" $P scripts/release_gate.py --model size-ladder \
      --size-ladder-record-lever SDPA_FUSED_LARGE_S --size-ladder-models "$m" || continue
  $P perf/ttx_a3/ladder_reason.py "$frag" p300c >> "$OUT/reasons.log" 2>&1
  $P scripts/release_gate.py --model size-ladder --size-ladder-fill-reasons \
      --size-ladder-models "$m" >> "$OUT/reasons.log" 2>&1
  ladder_with_retries "$m"
done
log "LADDER_CAMPAIGN_DONE card=$CARD models=$*"
