#!/bin/bash
# run.sh <chip> <out> <tree> <arm> <inputs> <warm> <seed> [stall_s]: one Protenix-v2 bench.py run armed to catch a hang
# live. Inspector RPC + a 300 s Metal op timeout run hangcap.sh against the hung process; a stall watch (no new rep and
# no log line for stall_s, default 1200) runs it too, for a hang Metal does not see (host side, or an op that never
# times out). After a dump the run is stopped (SIGINT, then SIGTERM) and the card reset (each qb1 p150a is its own board).
# A device init failure before the first fold (card left dirty by an earlier job) resets the card and retries once.
# Writes <out>/b/bench.jsonl, <out>/log, <out>/aiclk.tsv, <out>/dump/*, <out>/result (ok | hang <why> | rc <n>).
c=$1; out=$2; tree=$3; arm=$4; inp=$5; warm=$6; seed=$7; stall=${8:-1200}
H=$(dirname $(readlink -f $0)); V=$(ls -dt $HOME/scratch/shipflow/venv-*/ | head -1)
rm -rf $out; mkdir -p $out/insp; echo $V > $out/venv; cd $tree
reset() { timeout 300 $HOME/.local/bin/tt-smi -r $c >> $out/log 2>&1; echo "reset rc=$? $(date -u +%FT%TZ)"; }
# A card a capped job left dirty fails device init before any fold: reset it and start again, once.
for try in 1 2; do
  env PYTHONPATH=$tree TT_VISIBLE_DEVICES=$c TT_METAL_LOGS_PATH=$out/insp TT_METAL_INSPECTOR=1 TT_METAL_INSPECTOR_RPC=1 \
    TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:5008$c TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1 \
    TT_METAL_OPERATION_TIMEOUT_SECONDS=300 TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="bash $H/hangcap.sh $c $out optimeout" \
    ${V}bin/python perf/spd/bench.py --out $out/b --chip $c --model protenix-v2 --arm $arm --inputs $inp --seed $seed \
    --warm $warm > $out/log 2>&1 &
  P=$!; echo $P > $out/bench.pid
  ( while kill -0 $P 2>/dev/null; do n=$(ls -l /proc/$P/fd 2>/dev/null | grep -o 'tenstorrent/[0-9]*' | head -1 | cut -d/ -f2)
    [ -n "$n" ] && echo "$(date +%s) $n $(cat /sys/class/tenstorrent/tenstorrent!$n/tt_aiclk)"; sleep 2; done ) > $out/aiclk.tsv &
  while kill -0 $P 2>/dev/null; do sleep 20
    last=$(stat -c %Y $out/log $out/b/bench.jsonl 2>/dev/null | sort -n | tail -1)
    [ -d $out/dump/.taken ] || [ $(( $(date +%s) - ${last:-0} )) -lt $stall ] || bash $H/hangcap.sh $c $out stall
    if [ -s $out/dump/why ] && grep -q done $out/dump/why; then
      kill -INT $P; for i in $(seq 120); do kill -0 $P 2>/dev/null || break; sleep 1; done
      kill -0 $P 2>/dev/null && kill -TERM $P; break; fi
  done
  wait $P; rc=$?; sleep 3
  grep -q '"ev": "rep"' $out/b/bench.jsonl 2>/dev/null || ! grep -q "failed to initialize FW\|ARC core failed" $out/log \
    || [ $try = 2 ] || { echo "initfail $(reset)" >> $out/initfail; rm -rf $out/dump $out/b; continue; }
  break
done

if [ -d $out/dump/.taken ]; then echo "hang $(head -1 $out/dump/why)" > $out/result
  reset >> $out/result
elif [ $rc = 0 ]; then echo ok > $out/result; else echo "rc $rc" > $out/result; fi
