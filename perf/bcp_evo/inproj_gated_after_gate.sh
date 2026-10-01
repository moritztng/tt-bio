#!/bin/bash
# Stage 14 op bench on card 3 once the release gate (pid $1) has exited.
cd /home/ttuser/.coworker/wt/bcp-evo
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "bench start $(date -u +%FT%TZ)" >> perf/bcp_evo/out/inproj_gated_chain.txt
TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcp-evo \
  timeout 900 /home/ttuser/bcx_e2e_venv/bin/python -u perf/bcp_evo/inproj_gated_bench.py \
  --s 1,2 --reps 20 > perf/bcp_evo/out/inproj_gated_bench.log 2>&1
echo "bench rc $? $(date -u +%FT%TZ)" >> perf/bcp_evo/out/inproj_gated_chain.txt
