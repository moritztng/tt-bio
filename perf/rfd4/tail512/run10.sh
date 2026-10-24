#!/bin/bash
# tail512 run10: ceiling of NOCR in dedicated mode. Writer on NOC0 (dmcfg 1) with the stage reads gone (dm 2) proxies
# "stage reads issued on NOC0 by the writer"; with and without the sigmoid (abl 16). No dynamic-NOC leg (run9: one
# slows every dedicated program after it in the process).
#   bash perf/rfd4/tail512/run10.sh <chip> <outdir>
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run10: NOCR dedicated ceiling" \
  ";dm=2;dmcfg=1,dm=2;abl=16;abl=16,dm=2;abl=16,dmcfg=1;abl=16,dmcfg=1,dm=2;abl=91,dmcfg=1,dm=2;abl=91,dm=2"
