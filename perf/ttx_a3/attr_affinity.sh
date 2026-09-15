#!/usr/bin/env bash
# Attribute the boltz2-affinity perf red to the lever or to the box.
#
# The perf arm read 0.02413 -> 0.01063 affinities/s, -55.9 %, FAIL. Two facts make that number
# suspect before any control is run:
#
#   1. benchlock ACQUIRED at loadavg 1.82 and RELEASED at 10.58. benchlock checks the load once, at
#      acquire, so a sibling campaign that ramps up mid-measurement is invisible to it -- the
#      known one-shot-check blind spot. The three reps read 82, 94 and 102 s, a 24 % spread, which
#      is what a box filling up under you looks like.
#   2. The input is affinity_fkg.yaml, FKBP12+SB3 at L107. The route is gated strictly above 1024
#      tokens. At 107 residues it cannot fire, and the census counter can prove that rather than
#      it being argued.
#
# Neither fact is a control, so: off/on/off as separate processes against the same tree, each arm
# its own perf_regression run under benchlock, with loadavg recorded at BOTH ends of every arm so a
# mid-run ramp cannot hide again.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate2/affinity_attr"
PROG="$OUT/progress"
CARD=0
P=/home/ttuser/tt-bio-dev/env/bin/python3
BENCH=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

rep() {  # $1 = name, $2 = on|off
  local name="$1" arm="$2"
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  local ev=() l0 l1
  [ "$arm" = off ] && ev=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  l0=$(cut -d' ' -f1 /proc/loadavg)
  log "$name START arm=$arm load0=$l0"
  env "${ev[@]}" TT_VISIBLE_DEVICES=$CARD timeout 2700 \
    bash "$BENCH" "worker:ttx-a3-sdpa-ship-remerge" -- \
    "$P" scripts/perf_regression.py --model boltz2-affinity > "$OUT/$name.log" 2>&1
  local rc=$?
  l1=$(cut -d' ' -f1 /proc/loadavg)
  local got
  got=$(grep -oE "boltz2-affinity affinities/s +[0-9.]+ +[0-9.]+ +[-+][0-9.]+%" "$OUT/$name.log" | tail -1)
  log "$name rc=$rc arm=$arm load0=$l0 load1=$l1 | ${got:-no-row}"
}

rep aff_off1 off
rep aff_on1  on
rep aff_off2 off
rep aff_on2  on
log "AFFINITY_ATTR_DONE"
