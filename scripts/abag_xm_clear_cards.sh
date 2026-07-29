#!/usr/bin/env bash
# Stop this host's Tier-A generation drivers and release every card they were holding.
#
# Why this needs to be a script rather than a pkill: killing a driver does NOT kill the
# `tt_bio.main predict` it spawned, and killing that predict does not kill ITS multiprocessing
# worker child. The worker is what actually holds the card -- it takes an exclusive flock in
# tt_bio/device_lease.py -- and once orphaned it reparents to PPID 1 and keeps the lock for as
# long as it lives. A later fold on that card then waits 120 s on the flock and dies with
# DeviceInUseError, which the Rich display reports only as "all local workers exited".
#
# That is not hypothetical: on 2026-07-27 a driver killed at 20:28 left one such orphan on card
# 3, and every fold on that card failed for the next ~80 minutes. It reads exactly like dead
# hardware -- reset-proof, telemetry clean -- and it is not.
#
# The sweep therefore matches on the INHERITED environment (TT_VISIBLE_DEVICES) across all of
# this user's processes, not on a command line. The mp worker's argv is
# `python3 -c from multiprocessing.spawn import spawn_main; ...`, which no cmdline pattern for
# "predict" or "abag_xm" will ever match; the env var is the only thing that survives the spawn.
#
#   Usage: scripts/abag_xm_clear_cards.sh [cards]     cards default "0 1 2 3"
set -u
CARDS="${1:-0 1 2 3}"

holders() {  # holders <card> -> pids of any process of ours that inherited TT_VISIBLE_DEVICES=<card>
  local card="$1" p env
  for p in $(pgrep -u "$(id -u)" '' 2>/dev/null); do
    # 2>/dev/null must come BEFORE the input redirect: redirections are processed left to
    # right, so a failing "< /proc/<pid>/environ" (process already exited, or not ours) is
    # reported by the shell before a later stderr redirect can suppress it.
    env=$(2>/dev/null tr '\0' '\n' < "/proc/$p/environ" | grep -c "^TT_VISIBLE_DEVICES=${card}$") || true
    [ "${env:-0}" = "1" ] && echo "$p"
  done
  return 0
}

for p in $(pgrep -f 'abag_xm_generate.py' 2>/dev/null); do
  echo "kill driver $p"; kill -9 "$p" 2>/dev/null
done
sleep 3

for card in $CARDS; do
  for pass in 1 2 3; do
    pids=$(holders "$card")
    [ -z "$pids" ] && break
    for p in $pids; do
      echo "card $card: kill holder $p ($(ps -o args= -p "$p" 2>/dev/null | cut -c1-60))"
      kill -9 "$p" 2>/dev/null
    done
    sleep 3   # a predict can hand off to a fresh worker between passes
  done
done

rc=0
for card in $CARDS; do
  pids=$(holders "$card")
  if [ -n "$pids" ]; then echo "card $card: STILL HELD by $pids"; rc=1; else echo "card $card: free"; fi
done
# The lease dir should be empty once every holder is gone; a leftover file whose flock nobody
# holds is harmless (the next opener reclaims it) but is worth seeing.
echo "lease dir: $(ls -A /tmp/tt-bio-device-leases 2>/dev/null | tr '\n' ' ')"
exit $rc
