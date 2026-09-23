#!/usr/bin/env bash
# of3t-msaamp Addendum 1: the 64-token crop. Stage 2 reference from the f64 crop arm, then
# upstream bf16/fp32 and ours at 64, each behind a green host_quiet.py.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msaamp
cd "$W" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python; S=/home/ttuser/of3t_msaamp
HQ=perf/c12_orchestrator/pair_guard/host_quiet.py
gate() { local i; for i in $(seq 1 60); do python3 $HQ --quiet && { echo "host_quiet GREEN before $1 load[$(cut -d' ' -f1-3 /proc/loadavg)] $(date -u +%FT%TZ)" | tee -a perf/of3t_msaamp/HOST_QUIET.txt; return 0; }; sleep 10; done; echo "host_quiet RED before $1"; return 1; }
$PY perf/of3t_msaamp/make_crop.py --src $S/crop64_s1.pt --out $S/crop64.pt \
  --f64-dump $S/grads_crop64_f64.pt --f64-z $S/grads_crop64_f64_z.pt || exit 1
export BND=$S/crop64.pt
for spec in "f64 crop64_f64" "bf16auto crop64_bf16auto" "f32 crop64_f32"; do
  gate "$spec" || exit 1
  bash perf/of3t_msaamp/run_arm.sh $spec 2>&1 | grep -E "^===|HARD|real-block"
done
gate crop64_ours || exit 1
bash perf/of3t_msaamp/run_ours.sh crop64_ours 2>&1 | grep -E "^===|FWD|mass-weighted|worst"
$PY perf/of3t_msaamp/summarize.py crop64_ $S/crop64.pt
