#!/bin/bash
# bcp-land pass 2: the composed round on the no-flags default (lnbw on, from bcp-device) against
# the same tree with lnbw taken out, ABBAAB on one card, then the split.
cd "$(dirname "$0")/../.."
o=perf/bcp_land/out; mkdir -p $o
echo "compose start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) card $TT_VISIBLE_DEVICES" > $o/compose.log
off=TT_BIO_LNBW_FUSED=0
perf/bcp_land/sit.sh 9 "off1:auto:$off on1:auto on2:auto off2:auto:$off off3:auto:$off on3:auto" >> $o/compose.log 2>&1
~/bcx_e2e_venv/bin/python3 perf/bcp_device/split.py $o/off1 $o/on1 $o/on2 $o/off2 $o/off3 $o/on3 \
    > perf/bcp_land/split_compose.txt 2>&1
echo "compose done $(date -u +%FT%TZ)" >> $o/compose.log
