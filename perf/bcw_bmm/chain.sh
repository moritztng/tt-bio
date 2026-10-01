#!/bin/bash
# The sweep on card 0. BindCraft 2 pads the binder to a multiple of 32 before adding the target,
# so the seam axis is pad32(target + pad32(binder)): each binder sits in the one 32-wide binder
# bucket that lands its target on the axis named. 800 (hHSA 163), 768 (hHSA 150) and a first 288
# pair ran under earlier lists; the 288 pair runs again here with gradient digests.
wt=/home/ttuser/.coworker/wt/bcw-bmm
$wt/perf/bcw_bmm/sweep.sh 0 $wt "hPDL1:150:288"
$wt/perf/bcw_bmm/sweep.sh 0 $wt/perf/bcw_bmm/main_tree "hPDL1:150:288"
$wt/perf/bcw_bmm/sweep.sh 0 $wt "hPDL1:180:320 hPDL1:210:352 hIL7RA:150:384 hIL7RA:180:416 hIL7RA:210:448 hIL7RA:240:480 hIL2R:90:512 hIL2R:110:544 hIL2R:150:576 hIL2R:180:608 hIL2R:210:640 hHSA:60:672 hHSA:90:704 hHSA:110:736 hHSA:210:832 hHSA:240:864"
echo "=== CHAIN DONE $(date -u +%FT%TZ)" >> $wt/perf/bcw_bmm/out/sweep/card0.log
