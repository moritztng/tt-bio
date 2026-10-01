#!/bin/bash
# abb3 re-record on the bcp-evo merge tree, card 3. Raw outputs under perf/bcp_evo/out/abb3/.
cd ~/.coworker/wt/bcp-evo
O=perf/bcp_evo/out/abb3; mkdir -p $O
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcp-evo PYTHONPATH=$PWD
PY=~/tt-bio-dev/env/bin/python3
echo "start $(date -u +%FT%TZ) $(git rev-parse HEAD)" > $O/run.txt
for sigma in 0.05 0.3; do for tokens in 64 256; do
  echo "=== sigma=$sigma, $tokens tokens, 8 blocks, batch 2 ===" >> $O/model_gate.txt
  timeout 1800 $PY scripts/abb3_port/model_gate.py --tokens $tokens --sigma $sigma >> $O/model_gate.txt 2>&1
  echo "rc=$?" >> $O/model_gate.txt
done; done
echo "== --micro 8 --accumulate 8 --steps 1 --warmup 1 ==" >> $O/step_gate.txt
timeout 1800 $PY scripts/abb3_port/step_gate.py --micro 8 --accumulate 8 --steps 1 --warmup 1 >> $O/step_gate.txt 2>&1; echo "rc=$?" >> $O/step_gate.txt
echo "== default: --micro 4 --accumulate 16 --steps 10 --warmup 1 ==" >> $O/step_gate.txt
timeout 3600 $PY scripts/abb3_port/step_gate.py >> $O/step_gate.txt 2>&1; echo "rc=$?" >> $O/step_gate.txt
echo "end $(date -u +%FT%TZ)" >> $O/run.txt
