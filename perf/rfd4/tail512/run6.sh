#!/bin/bash
# tail512 run6: which compute stage the writer's g / z reads collide with: reads only (dm 11) against no DM (dm 15)
# beside compute with stages skipped (abl bits 1 LN, 2 GEMM, 8 epilogue, 16 sigmoid).
#   bash perf/rfd4/tail512/run6.sh <chip> <outdir>
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run6: read collision" \
  "abl=16,dm=11;abl=17,dm=11;abl=17,dm=15;abl=18,dm=11;abl=18,dm=15;abl=24,dm=11;abl=24,dm=15;abl=26,dm=11;abl=26,dm=15;abl=25,dm=11;abl=25,dm=15;abl=27,dm=11;abl=91,dm=11;dm=11;dm=15"
