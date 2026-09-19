#!/bin/bash
# Pull every leg down to the numbers this row reports: engine runtime + depth at the model
# boundary out of results.json, structure verdict out of the shared instrument, power extremes
# out of the forward-progress trace.
set -u
D=/home/cust-team/mthuening/ptxcovwh
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
CHK=/home/cust-team/mthuening/obcovwh/eng-main/perf/wh-correctness/check_structure.py
for t in m1024 p1024 m1024s7; do
  echo "===== $t ====="
  cat $D/logs/$t.run
  R=$(ls $D/out/$t/protenix_results_*/results.json 2>/dev/null | head -1)
  if [ -n "$R" ]; then TT_VISIBLE_DEVICES= $PY $D/rj.py "$R"; else echo "  NO results.json"; fi
  C=$(ls $D/out/$t/protenix_results_*/structures/*.cif 2>/dev/null | head -1)
  if [ -n "$C" ]; then
    echo "  cif: $C"
    echo "  md5: $(md5sum "$C" | cut -d' ' -f1)"
    TT_VISIBLE_DEVICES= $PY $CHK "$C" --conf "$R" 2>&1 | tail -22 | sed 's/^/    /'
  else echo "  NO cif"; fi
  echo "  power W min/max: $(awk -F'P=' '{print $2/1000000}' $D/logs/$t.pwr | sort -n | sed -n '1p;$p' | tr '\n' '/')"
  echo "  absorbed: $(grep -c 'Out of Memory' $D/logs/$t.log) DRAM refusal, $(grep -c 'Statically allocated circular buffers' $D/logs/$t.log) L1-CB throw"
  grep -oE 'allocate [0-9]+ B DRAM' $D/logs/$t.log | sort | uniq -c | sed 's/^/    /'
done
