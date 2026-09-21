#!/bin/bash
# MSA depth arms, one tt-bio predict per rung, same card and same command as run1.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
cd "$WT"
CARD=${CARD:-2}
source /home/ttuser/tt-bio-dev/env/bin/activate 2>/dev/null || true
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
nohup python3 perf/pvxrerun/clk_sample.py "$CARD" "$WT/perf/pvxrerun/depth_aiclk.jsonl" >/dev/null 2>&1 &
echo $! > perf/pvxrerun/depth_clk.pid
for tag in d2048 d3553 d9947; do
  rm -rf "perf/pvxrerun/out_$tag"; mkdir -p "perf/pvxrerun/out_$tag"
  date -u +"START $tag %Y-%m-%dT%H:%M:%SZ" >> perf/pvxrerun/depth.log
  tt-bio predict "$WT/perf/pvxrerun/in_$tag" --model protenix-v2 --diffusion_samples 5 \
    --sampling_steps 200 --recycling_steps 10 --msa_dir "$WT/perf/pvxrerun/msa_$tag" \
    --msa_cache_only --seed 101 --override --out_dir "$WT/perf/pvxrerun/out_$tag" \
    >> perf/pvxrerun/depth.log 2>&1
done
kill "$(cat perf/pvxrerun/depth_clk.pid)" 2>/dev/null
