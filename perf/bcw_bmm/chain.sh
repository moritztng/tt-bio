#!/bin/bash
# The whole sweep on card 0: the 768 fix first, the 288 bit-exact pair (branch, then main), then
# every other axis 320..864.
wt=/home/ttuser/.coworker/wt/bcw-bmm
s=$wt/perf/bcw_bmm/sweep.sh
$s 0 $wt "hHSA:163:768 hPDL1:152:288"
$s 0 $wt/perf/bcw_bmm/main_tree "hPDL1:152:288"
$s 0 $wt "hPDL1:184:320 hPDL1:216:352 hPDL1:248:384 hCA2:138:416 hCA2:170:448 hCA2:202:480 hIL2R:81:512 hIL2R:113:544 hIL2R:145:576 hIL2R:177:608 hIL2R:209:640 hHSA:67:672 hHSA:99:704 hHSA:131:736 hHSA:195:800 hTF:132:832 hTF:164:864"
echo "=== CHAIN DONE $(date -u +%FT%TZ)" >> $wt/perf/bcw_bmm/out/sweep/card0.log
