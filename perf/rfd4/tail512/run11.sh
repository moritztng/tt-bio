#!/bin/bash
# tail512 run11: lever NOCR in dedicated mode. T2_WSTAGE moves the stage reads into the writer (c_24 two units deep);
# with dmcfg 1 the writer (all DRAM reads, the writes) is on NOC0 and the reader (gather only) on NOC1.
# Full call and without the sigmoid (abl 16, the PSIG ceiling).
#   bash perf/rfd4/tail512/run11.sh <chip> <outdir>
L=""
for a in "" "abl=16,"; do L="$L;${a%,};${a}wstage=1;${a}dmcfg=1,wstage=1;${a}dmcfg=1"; done
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run11: NOCR dedicated (WSTAGE)" "${L#;}"
