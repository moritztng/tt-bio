#!/bin/bash
# tail512 run7: does the writer's g / z read collision with compute follow its NOC or its processor? The two DM kernels
# with NOCs swapped (dmcfg 1), processors swapped (2), both (5) and the defaults spelled out (4), weight multicast off
# in every leg (mcast 0, bit-identical), on the full call and on reads + compute without the sigmoid (abl 16, dm 11).
#   bash perf/rfd4/tail512/run7.sh <chip> <outdir>
L="mcast=0"
for c in 4 1 2 5; do L="$L;mcast=0,dmcfg=$c"; done
L="$L;abl=16,dm=11,mcast=0"
for c in 4 1 2 5; do L="$L;abl=16,dm=11,mcast=0,dmcfg=$c"; done
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run7: DM placement" "$L"
