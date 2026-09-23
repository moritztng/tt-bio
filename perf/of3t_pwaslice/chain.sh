#!/usr/bin/env bash
# of3t-pwaslice chain on card 2, 64-token batch: fix arm PW64F, base arm PW64B, then the OpenFold3
# fold digest base vs fix. The guard control on the fix already ran (GUARD_CONTROL_FIX.json).
W=/home/ttuser/.coworker/wt/of3t-pwaslice; S=/home/ttuser/of3t_pwaslice; L=$S/chain.log
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][2]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain.txt 2>&1 &
M=$!
bash perf/of3t_pwaslice/devarm.sh PW64F $W >> $L 2>&1
bash perf/of3t_pwaslice/devarm.sh PW64B $S/base >> $L 2>&1
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
