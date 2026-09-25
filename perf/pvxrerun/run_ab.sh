#!/bin/bash
# One engine arm of the contention-cancelling A/B: the same 672 fixture twice, on one card,
# out of an extracted source tree. Both arms are launched together so they see the same host
# load, which is the only way to separate "the host is busy" from "the engine changed" while
# the fleet is working.
#   run_ab.sh <tree-dir> <card> <tag>
# `python3 -m tt_bio.main` resolves tt_bio from cwd. The `tt-bio` console script does NOT --
# it resolves from /home/ttuser/tt-bio-dev whatever the cwd is -- so the module form is the
# only way to pin which tree is measured.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
TREE=$1; CARD=$2; TAG=$3
RUN=$WT/perf/pvxrerun/$TAG
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
cp "$WT/perf/pvxrerun/yaml/b78_tok658_bk672.yaml" "$RUN/in/01_b78_tok658_bk672.yaml"
cp "$WT/perf/pvxrerun/yaml/b78_tok658_bk672.yaml" "$RUN/in/02_b78_tok658_bk672.yaml"
cd "$TREE"
nohup /home/ttuser/tt-bio-dev/env/bin/python3 "$WT/perf/pvxrerun/clk_sample.py" "$CARD" "$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
nohup bash -c "while :; do echo \"\$(date -u +%H:%M:%S) \$(cat /proc/loadavg)\" >> $RUN/loadavg.log; sleep 10; done" >/dev/null 2>&1 &
echo $! > "$RUN/load.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=1,2
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
echo "TREE=$TREE CARD=$CARD" >> "$RUN/run.log"
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict "$RUN/in" --model protenix-v2 \
  --diffusion_samples 5 --sampling_steps 200 --recycling_steps 10 \
  --msa_dir "$WT/perf/pvxrerun/msa" --msa_cache_only --seed 101 --override \
  --out_dir "$RUN/out" >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" "$(cat "$RUN/load.pid")" 2>/dev/null
