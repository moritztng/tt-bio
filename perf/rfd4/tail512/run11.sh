#!/bin/bash
# tail512 run11: dedicated-mode NOCR and the gather issue rate. T2_WSTAGE: the writer reads the stage (c_24 two units
# deep), the reader only gathers; with dmcfg 1 every DRAM read is on NOC0. T2_GWR: the gather loopback copies also go
# out through the write command buffer (1 all, 2 the right faces). Full call, DM only without the stage reads
# (abl 91, dm 2: run10 1.59 floor) and without the sigmoid.
#   bash perf/rfd4/tail512/run11.sh <chip> <outdir>
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run11: WSTAGE, GWR" \
  ";gwr=2;gwr=1;wstage=1;dmcfg=1,wstage=1;abl=91,dm=2;abl=91,dm=2,gwr=2;abl=91,dm=2,gwr=1;abl=16;abl=16,gwr=2;abl=16,dmcfg=1,wstage=1"
