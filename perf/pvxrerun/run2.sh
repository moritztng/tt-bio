#!/bin/bash
# Re-run of the customer command on qb1 card 1 (p150a, their board class).
# Queue: the three buckets the previous pass measured, then a 672 repeat for the A/A
# floor, then the 736 bucket last -- 736 is the shape that ground for 1h40m on qb2 and
# putting it last keeps a wedge from costing the other three numbers.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
cd "$WT"
CARD=${CARD:-1}
RUN=perf/pvxrerun/run2
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
i=0
for stem in b78_tok658_bk672 b108_tok688_bk704 b168_tok748_bk768 \
            b78_tok658_bk672 b108_tok688_bk704 b168_tok748_bk768 \
            b140_tok720_bk736 b140_tok720_bk736; do
  i=$((i+1))
  cp perf/pvxrerun/yaml/$stem.yaml "$RUN/in/$(printf %02d $i)_$stem.yaml"
done
nohup /home/ttuser/tt-bio-dev/env/bin/python3 perf/pvxrerun/clk_sample.py "$CARD" "$WT/$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
/home/ttuser/tt-bio-dev/env/bin/tt-bio predict "$WT/$RUN/in" --model protenix-v2 --diffusion_samples 5 \
  --sampling_steps 200 --recycling_steps 10 --msa_dir "$WT/perf/pvxrerun/msa" \
  --msa_cache_only --seed 101 --override --out_dir "$WT/$RUN/out" \
  >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" 2>/dev/null
