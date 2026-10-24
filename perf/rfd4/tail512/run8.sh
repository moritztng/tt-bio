#!/bin/bash
# tail512 run8: lever NOCR. T2_XNOC moves DRAM reads to NOC0 under DM_DYNAMIC_NOC: 2 = the writer g / z reads (form a),
# 3 = all writer traffic, dmcfg 1 + xnoc 5 = reader on NOC1 (gather) with its stage reads on NOC0, writer on NOC0 with
# its writes on NOC1 (form b). Each on the full call and without the sigmoid (abl 16, the PSIG ceiling).
#   bash perf/rfd4/tail512/run8.sh <chip> <outdir>
L=""
for a in "" "abl=16,"; do L="$L;${a%,};${a}xnoc=2;${a}xnoc=3;${a}dmcfg=1,xnoc=5"; done
exec bash "$(dirname "$0")/legs.sh" "$1" "$2" "tail512 run8: NOCR" "${L#;}"
