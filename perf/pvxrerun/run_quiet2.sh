#!/bin/bash
# The one measurement this row still owes: the 736 bucket on a QUIET p150a, plus the 672
# bucket beside it as a cross-calibrator against the previous pass benchlocked 157.862 s.
# Run under benchlock, which waits for the box to go quiet first. loadavg is sampled every
# 10 s regardless, so the "quiet" label on each fold comes from measured samples and not
# from the lock having promised it.
set -u
WT=/home/ttuser/.coworker/wt/pvx-custchart-rerun
CARD=1
RUN=$WT/perf/pvxrerun/runq2
rm -rf "$RUN"; mkdir -p "$RUN/in" "$RUN/out"
i=0
for stem in b78_tok658_bk672 b78_tok658_bk672 b140_tok720_bk736 b140_tok720_bk736; do
  i=$((i+1)); cp "$WT/perf/pvxrerun/yaml/$stem.yaml" "$RUN/in/$(printf %02d $i)_$stem.yaml"
done
cd "$WT/perf/pvxrerun/new56"
nohup /home/ttuser/tt-bio-dev/env/bin/python3 "$WT/perf/pvxrerun/clk_sample.py" "$CARD" "$RUN/aiclk.jsonl" >/dev/null 2>&1 &
echo $! > "$RUN/clk.pid"
nohup bash -c "while :; do echo \"\$(date -u +%H:%M:%S) \$(cat /proc/loadavg)\" >> $RUN/loadavg.log; sleep 10; done" >/dev/null 2>&1 &
echo $! > "$RUN/load.pid"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:pvx-custchart-rerun
date -u +"START %Y-%m-%dT%H:%M:%SZ" > "$RUN/run.log"
echo "loadavg at start: $(cat /proc/loadavg)" >> "$RUN/run.log"
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict "$RUN/in" --model protenix-v2 \
  --diffusion_samples 5 --sampling_steps 200 --recycling_steps 10 \
  --msa_dir "$WT/perf/pvxrerun/msa" --msa_cache_only --seed 101 --override \
  --out_dir "$RUN/out" >> "$RUN/run.log" 2>"$RUN/run.err"
date -u +"END %Y-%m-%dT%H:%M:%SZ" >> "$RUN/run.log"
kill "$(cat "$RUN/clk.pid")" "$(cat "$RUN/load.pid")" 2>/dev/null
