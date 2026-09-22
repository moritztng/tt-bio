#!/bin/bash
# The benchlocked four-bucket run this row owes: all four token buckets folded twice in ONE
# process on ONE card on a measured-quiet host, off the customer's own command, read off
# tt-bio's own per-fold stdout line.
#
# Bucket order is a palindrome (672 704 736 768 768 736 704 672) so the first four folds pay
# each shape's kernel compilation and the last four are warm. If the run is cut short, the
# warm fold lost first is 672, the one bucket the previous pass already measured benchlocked
# three times.
set -u
WT=/home/moritz/.coworker/wt/pvx-custquote
CARD=0
RUN=$WT/perf/pvxrerun/quietpc
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
i=0
for stem in b78_tok658_bk672 b108_tok688_bk704 b140_tok720_bk736 b168_tok748_bk768 \
            b168_tok748_bk768 b140_tok720_bk736 b108_tok688_bk704 b78_tok658_bk672; do
  i=$((i+1)); cp "$WT/perf/pvxrerun/yaml/$stem.yaml" "$RUN/in/$(printf %02d $i)_$stem.yaml"
done
cd "$WT"
nohup /home/moritz/tt-bio/env/bin/python3 "$WT/perf/pvxrerun/clk_sample_pc.py" "$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
nohup bash -c "while :; do echo \"\$(date -u +%H:%M:%S) \$(cat /proc/loadavg)\" >> $RUN/loadavg.log; sleep 10; done" >/dev/null 2>&1 &
echo $! > "$RUN/load.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:pvx-custquote
{
  date -u +"START %Y-%m-%dT%H:%M:%SZ"
  echo "host: $(hostname)   loadavg at start: $(cat /proc/loadavg)"
  echo "tree: $(git rev-parse HEAD)   card: UMD $CARD"
} > "$RUN/run.log"
/home/moritz/tt-bio/env/bin/python3 -m tt_bio.main predict "$RUN/in" --model protenix-v2 \
  --diffusion_samples 5 --sampling_steps 200 --recycling_steps 10 \
  --msa_dir "$WT/perf/pvxrerun/msa" --msa_cache_only --seed 101 --override \
  --out_dir "$RUN/out" >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" "$(cat "$RUN/load.pid")" 2>/dev/null
