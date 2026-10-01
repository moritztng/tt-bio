#!/bin/bash
# Stage 14 op sweep: fidelity x x-buffering at the round shape and two edge shapes.
cd /home/ttuser/.coworker/wt/bcp-evo
for cfg in "HiFi4 2" "HiFi2 2" "HiFi4 1" "LoFi 2"; do
  set -- $cfg
  TT_BIO_INPROJ_GATED_FIDELITY=$1 TT_BIO_INPROJ_GATED_XBUF=$2 timeout 600 /home/ttuser/bcx_e2e_venv/bin/python -u     perf/bcp_evo/inproj_gated_bench.py --s 1,2 --reps 20 --out perf/bcp_evo/out/ipg_sweep_$1_x$2.json     > perf/bcp_evo/out/ipg_sweep_$1_x$2.log 2>&1
  echo "$1 x$2 rc $? $(date -u +%FT%TZ)" >> perf/bcp_evo/out/ipg_sweep_chain.txt
done
