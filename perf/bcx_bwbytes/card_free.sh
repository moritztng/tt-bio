#!/bin/bash
# Is a card actually free? Run this ON THE ORCHESTRATOR: it is the only host that can answer the
# fourth question. One line per CARD; exit 0 if any is takeable.
#
# Four conditions, each of which has on its own made a card look free when it was not:
#   1 no live holder on its device node          (fuser)
#   2 no lease whose pid is ALIVE                -- a lease can read released 2.1 s after acquire
#                                                   while its holder runs another hour
#   3 no LIVE cardblock                          -- hazard notes outlive the arm that caused them,
#                                                   AND a cardblock binds the DISPATCHER, not a
#                                                   detached arm: a blocked card has run for 75
#                                                   minutes under one. So the four conditions are
#                                                   independent, not ordered -- a blocked card can
#                                                   ALSO have a live holder, and all reasons are
#                                                   accumulated rather than short-circuited.
#   4 no live worker.sh ASSIGNED to host+card    -- the one that is easy to miss: an arm exits and
#                                                   releases its lease while the ROW is alive
#                                                   between stages and about to open again. Only
#                                                   the orchestrator's process table shows it, and
#                                                   on 2026-09-26 qb2 card 3 read free on all three
#                                                   of the other conditions while land-standing
#                                                   held the assignment.
set -u
D=${D:-$HOME/.coworker}
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE=$HERE/card_occupancy.py
free_any=1
for spec in "qb1:tt-quietbox" "qb2:tt-quietbox2"; do
  short=${spec%%:*}; host=${spec##*:}
  occ=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "ttuser@$host" python3 - < "$PROBE" 2>/dev/null) \
    || { echo "$short: unreachable"; continue; }
  while IFS='|' read -r card node holder leaselive; do
    [ -z "${card:-}" ] && continue
    why=""
    [ -n "$holder" ]    && why="$why holder=$holder"
    [ -n "$leaselive" ] && why="$why live-lease=$leaselive"
    [ -f "$D/state/cardblock-$short-$card" ] && why="$why cardblock"
    row=$(pgrep -af "worker\.sh [^ ]+ $short $card " 2>/dev/null | head -1 \
          | sed -E 's|.*worker\.sh ([^ ]+) .*|\1|')
    [ -n "$row" ] && why="$why assigned-to=$row"
    if [ -z "$why" ]; then
      echo "$short card $card (node $node): FREE"
      free_any=0
    else
      echo "$short card $card (node $node): busy --$why"
    fi
  done <<< "$occ"
done
exit $free_any
