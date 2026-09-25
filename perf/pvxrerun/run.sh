#!/bin/bash
# Re-run the customer's benchmark with their exact command. One tt-bio predict over a
# directory of targets; the per-fold time is read back off tt-bio's own stdout line.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
cd "$WT"
CARD=${CARD:-2}
RUN=perf/pvxrerun/run1
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
# queue order: the 736 bucket first (the one the previous pass could only interpolate),
# then 672 (which cross-calibrates against qb1's p150a), then 704 and 768.
i=0
for stem in b140_tok720_bk736 b140_tok720_bk736 b140_tok720_bk736 \
            b78_tok658_bk672 b78_tok658_bk672 b78_tok658_bk672 \
            b108_tok688_bk704 b108_tok688_bk704 \
            b168_tok748_bk768 b168_tok748_bk768 \
            b108_tok688_bk704 b168_tok748_bk768; do
  i=$((i+1))
  cp perf/pvxrerun/yaml/$stem.yaml "$RUN/in/$(printf %02d $i)_$stem.yaml"
done
nohup python3 perf/pvxrerun/clk_sample.py "$CARD" "$WT/$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
source /home/ttuser/tt-bio-dev/env/bin/activate 2>/dev/null || true
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
tt-bio predict "$WT/$RUN/in" --model protenix-v2 --diffusion_samples 5 \
  --sampling_steps 200 --recycling_steps 10 --msa_dir "$WT/perf/pvxrerun/msa" \
  --msa_cache_only --seed 101 --override --out_dir "$WT/$RUN/out" \
  >> "$RUN/run.log" 2>&1
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" 2>/dev/null
