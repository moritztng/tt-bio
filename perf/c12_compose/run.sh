#!/usr/bin/env bash
# The composed fold session, benchlocked, with the box checked BEFORE a card is opened.
#
# Three things this wrapper exists for, all of them things that have already cost the campaign a
# session:
#
# 1. benchlock's exit 75 is contention between benchlock USERS only. Host load alone does not
#    produce it: the script takes the flock, sits in its load-wait loop, and after
#    BENCHLOCK_LOAD_WAIT_S prints "Proceeding, RECORD THIS" and runs the command anyway. A caller
#    that waits it out on a loaded box gets a contaminated number, not a refusal. So the load is
#    checked here, up front, and the run is REFUSED rather than contaminated.
# 2. A C12 row sitting in a wait loop holding a /dev/tenstorrent fd starves the v0.9.0 release
#    chain, whose arms 2-4 are quiet=1 and whose busy_holders() counts any outside process holding
#    one. Nothing here opens a card until the box is already quiet.
# 3. If benchlock proceeds anyway, the log is grepped for its own warning and the session is marked
#    contaminated in the JSON's sibling .verdict file rather than quietly reported.
#
#   run.sh <tag> [extra fold_compose.py args...]
set -u
TAG="${1:?usage: run.sh <tag> [fold_compose.py args...]}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out"; mkdir -p "$OUT"
CARD="${CARD:-2}"
MAXLOAD="${MAXLOAD:-3.0}"

load=$(awk '{print $1}' /proc/loadavg)
holder=$(cat "$HOME/.coworker/state/benchlock" 2>/dev/null || true)
busy=$(ls -l /proc/*/fd 2>/dev/null | grep -c "tenstorrent/$CARD" || true)

echo "run.sh $TAG: loadavg=$load maxload=$MAXLOAD benchlock_holder='${holder:-none}' card${CARD}_fds=$busy"
if [ -n "$holder" ]; then
  echo "run.sh: REFUSED, benchlock is held by: $holder" >&2
  echo "run.sh: a sibling's timed run is the blocker before the load is. Retry later." >&2
  exit 75
fi
if awk -v l="$load" -v m="$MAXLOAD" 'BEGIN{exit !(l>m)}'; then
  echo "run.sh: REFUSED, loadavg $load is above $MAXLOAD." >&2
  echo "run.sh: benchlock would NOT refuse this -- it would wait, then proceed anyway and hand back" >&2
  echo "run.sh: a contaminated number. Release the card and defer instead." >&2
  exit 75
fi
if [ "$busy" -gt 0 ]; then
  echo "run.sh: REFUSED, /dev/tenstorrent/$CARD already has $busy fd(s) open by another process." >&2
  exit 75
fi

LOG="$OUT/${TAG}.log"
export BENCHLOCK_WAIT_S="${BENCHLOCK_WAIT_S:-180}"
export BENCHLOCK_LOAD_WAIT_S="${BENCHLOCK_LOAD_WAIT_S:-600}"
cd "$REPO" || exit 1
TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:c12-compose-fold \
  "$HOME/.coworker/scripts/benchlock.sh" c12-compose-fold -- \
  python3 "$HERE/fold_compose.py" --out "$OUT/${TAG}.json" --cifs "$OUT/${TAG}_cifs" "$@" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "run.sh: benchlock+fold exit $rc" | tee -a "$LOG"
if grep -q "RECORD THIS" "$LOG"; then
  echo "CONTAMINATED: benchlock waited out its load window and proceeded anyway" > "$OUT/${TAG}.verdict"
  echo "run.sh: benchlock PROCEEDED ANYWAY after its load wait -- session marked contaminated" >&2
else
  echo "clean: benchlock did not have to wait out its load window" > "$OUT/${TAG}.verdict"
fi
exit "$rc"
