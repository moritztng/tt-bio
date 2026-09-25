#!/bin/bash
# Control arm: the 672 bucket three times in one process on qb1 card 2, so the two warm
# folds are a real A/A pair and the warm number can be read against run2 card 1, which
# ran while land-standing was folding on card 0 at loadavg 8-15.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
cd "$WT"
CARD=2
RUN=perf/pvxrerun/run3
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
i=0
for stem in b78_tok658_bk672 b78_tok658_bk672 b78_tok658_bk672; do
  i=$((i+1)); cp perf/pvxrerun/yaml/$stem.yaml "$RUN/in/$(printf %02d $i)_$stem.yaml"
done
nohup /home/ttuser/tt-bio-dev/env/bin/python3 perf/pvxrerun/clk_sample.py "$CARD" "$WT/$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
nohup bash -c "while :; do echo \"\$(date -u +%H:%M:%S) \$(cat /proc/loadavg)\" >> $WT/$RUN/loadavg.log; sleep 10; done" >/dev/null 2>&1 &
echo $! > "$RUN/load.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=1,2
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
/home/ttuser/tt-bio-dev/env/bin/tt-bio predict "$WT/$RUN/in" --model protenix-v2 --diffusion_samples 5 \
  --sampling_steps 200 --recycling_steps 10 --msa_dir "$WT/perf/pvxrerun/msa" \
  --msa_cache_only --seed 101 --override --out_dir "$WT/$RUN/out" \
  >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" "$(cat "$RUN/load.pid")" 2>/dev/null
