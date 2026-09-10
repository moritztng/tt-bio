#!/bin/bash
# Quiet-CARD check for qb1 card 1 = UMD chip 1 = PCI 0000:41:00.0 = /dev/tenstorrent/2.
# (UMD id != /dev node on this box: UMD 0->node1, 1->node2, 2->node3, 3->node0.)
# No number is recorded unless this prints QUIET. Sibling cards are reported, not gated:
# five other tasks share qb1 tonight, each pinned to its own card.
NODE=/dev/tenstorrent/2
other=""
for p in $(lsof -t $NODE 2>/dev/null | sort -u); do
  other="$other $p($(ps -o comm= -p "$p" 2>/dev/null))"
done
la=$(cut -d" " -f1 </proc/loadavg)
co=""
for n in 0 1 3; do
  h=$(lsof -t /dev/tenstorrent/$n 2>/dev/null | sort -u | tr "\n" "," )
  [ -n "$h" ] && co="$co node$n:$h"
done
if [ -n "$other" ]; then
  echo "BUSY card1 loadavg=$la holders:$other"
  exit 1
fi
echo "QUIET card1 loadavg=$la no holders on $NODE; sibling-cards:${co:- none}"
