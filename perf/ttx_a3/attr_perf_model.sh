#!/usr/bin/env bash
# Attribute ONE perf_regression red to the lever or to the box: off/on/off/on, alternating,
# separate processes, same tree, the off arms with TT_BIO_SDPA_FUSED_LARGE_S=0.
#
#   attr_perf_model.sh <model> [reps] [first-arm: off|on]
#
# This replaces attr_affinity.sh and attr_esmc.sh, which were two near-copies differing only in a
# model name. Three perf cells needed this treatment in one gate run, so the model is DATA.
#
# Two things make it a real control rather than a reassuring ritual. The arms alternate, so a drift
# in the box over the run cannot be read as an arm difference -- which matters here because qb2 is
# resetting every 8 to 15 minutes with a co-tenant campaign on cards 1 and 2. And loadavg is
# recorded at BOTH ends of every arm: benchlock samples the load once, at acquire, so a sibling that
# ramps up mid-measurement is invisible to it, and that is exactly how the first affinity reading
# came out -55.9% with the load going 1.82 -> 10.58 underneath it.
#
# Read the result as an interval, not a point. If the two off arms straddle the on arms, the cell is
# not decidable on one draw on this host and the flag is not implicated either way.
#
# REVERSE THE ORDER before believing an arm-correlated result. Strict alternation starting with off
# puts every off arm in an odd position and every on arm in an even one, so any per-position effect
# -- a run that leaves the card slower for whatever comes next, a page cache that is cold on every
# other invocation -- is indistinguishable from a real arm difference. Running the same reps with
# `on` first separates them: a slowdown that follows the POSITION is an artefact, one that follows
# the ARM is real. esmc-6b needed exactly this: off/on/off/on read -1.1/-38.7/-1.1/-36.7 %, which
# is equally consistent with "the on arm is 1.58x slower" and with "every second run is 1.58x
# slower".
set -u
M="${1:?usage: attr_perf_model.sh <model> [reps] [first-arm]}"
REPS="${2:-2}"
FIRST="${3:-off}"
[ "$FIRST" = off ] && SECOND=on || SECOND=off
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate2/attr_$M"
PROG="$OUT/progress"
P=/home/ttuser/tt-bio-dev/env/bin/python3
BENCH=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"; touch "$PROG"
export PYTHONPATH="$WT" TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }
rep() {
  local name="$1" arm="$2"
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  local ev=() l0 l1 rc
  [ "$arm" = off ] && ev=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  l0=$(cut -d' ' -f1 /proc/loadavg)
  env "${ev[@]}" TT_VISIBLE_DEVICES=0 timeout 2700 bash "$BENCH" \
    "worker:ttx-a3-sdpa-ship-remerge" -- "$P" scripts/perf_regression.py --model "$M" \
    > "$OUT/$name.log" 2>&1
  rc=$?
  l1=$(cut -d' ' -f1 /proc/loadavg)
  log "$name rc=$rc arm=$arm load0=$l0 load1=$l1 | $(grep -oE "^$M *[a-z/]+ +[0-9.]+ +[0-9.]+ +[-+][0-9.]+%" "$OUT/$name.log" | tail -1)"
}
for i in $(seq 1 "$REPS"); do
  rep "${FIRST}${i}_first" "$FIRST"
  rep "${SECOND}${i}_second" "$SECOND"
done
log "ATTR_${M}_${FIRST}FIRST_DONE"
cat "$PROG"
