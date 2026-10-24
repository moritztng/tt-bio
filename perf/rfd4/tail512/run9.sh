#!/bin/bash
# tail512 run9: does a DM_DYNAMIC_NOC program slow the dedicated-mode programs after it? run8 read base 1.79 in its
# first round and 2.23 in every round after the first xnoc leg. Base and abl 16 before and after one xnoc leg.
#   bash perf/rfd4/tail512/run9.sh <chip> <outdir>
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run9: dynamic NOC aftereffect" ";abl=16;xnoc=2;mcast=0;abl=16,mcast=0;dmcfg=4" 1
