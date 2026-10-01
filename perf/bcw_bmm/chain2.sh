#!/bin/bash
# After the sweep: the 288 pair again with per-round gradient digests, branch then main.
wt=/home/ttuser/.coworker/wt/bcw-bmm
while pgrep -f "perf/bcw_bmm/chain.sh" > /dev/null; do sleep 10; done
$wt/perf/bcw_bmm/sweep.sh 0 $wt "hPDL1:150:288"
$wt/perf/bcw_bmm/sweep.sh 0 $wt/perf/bcw_bmm/main_tree "hPDL1:150:288"
echo "=== CHAIN2 DONE $(date -u +%FT%TZ)" >> $wt/perf/bcw_bmm/out/sweep/card0.log
