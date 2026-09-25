#!/bin/bash
# Reproduce the p300c 736 hang, with the transition trace on.
#   run_p300c.sh <tree-dir> <card> <tag>
# TT_BIO_TRANSITION_TRACE=sync drains the device queue BEFORE each line is written, so the
# LAST line in the trace is the call the device is stuck on and not wherever the host ran
# ahead to. That distinction is the whole point of running this arm at all.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
TREE=$1; CARD=$2; TAG=$3
RUN=$WT/perf/pvxrerun/$TAG
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
cp "$WT/perf/pvxrerun/yaml/b140_tok720_bk736.yaml" "$RUN/in/01_b140_tok720_bk736.yaml"
cd "$TREE"
nohup /home/ttuser/tt-bio-dev/env/bin/python3 "$WT/perf/pvxrerun/clk_sample.py" "$CARD" "$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
nohup bash -c "while :; do echo \"\$(date -u +%H:%M:%S) \$(cat /proc/loadavg)\" >> $RUN/loadavg.log; sleep 10; done" >/dev/null 2>&1 &
echo $! > "$RUN/load.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=2,3
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
export TT_BIO_TRANSITION_TRACE=sync
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
echo "TREE=$TREE CARD=$CARD" >> "$RUN/run.log"
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict "$RUN/in" --model protenix-v2 \
  --diffusion_samples 5 --sampling_steps 200 --recycling_steps 10 \
  --msa_dir "$WT/perf/pvxrerun/msa" --msa_cache_only --seed 101 --override \
  --out_dir "$RUN/out" >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" "$(cat "$RUN/load.pid")" 2>/dev/null
