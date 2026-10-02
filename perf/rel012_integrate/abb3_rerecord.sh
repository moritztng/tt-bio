#!/bin/bash
# abb3 re-record on the rel012 integration tree (wk/rel012-integrate), card $1. Same runs as
# perf/bcw_land/abb3_rerecord.sh; raw outputs under perf/rel012_integrate/out/abb3/.
cd "$(dirname "$0")/../.."
card=$1; O=${O:-perf/rel012_integrate/out/abb3}; mkdir -p $O
while fuser /dev/tenstorrent/$card >/dev/null 2>&1; do sleep 20; done
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rel012-integrate PYTHONPATH=$PWD
PY=~/tt-bio-dev/env/bin/python3
echo "start $(date -u +%FT%TZ) $(git rev-parse HEAD)" > $O/run.txt
( while :; do echo "$(date -u +%s) $(cat /sys/class/tenstorrent/tenstorrent!$card/tt_aiclk 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $O/aiclk.txt &
ck=$!; trap "kill $ck 2>/dev/null" EXIT
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
