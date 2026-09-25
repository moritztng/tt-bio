#!/usr/bin/env bash
# of3t-msaamp: our arm with the orientation fix, plus the legacy control that must reproduce the
# banked number exactly. Each behind a green host_quiet.py.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msaamp
cd "$W" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python; S=/home/ttuser/of3t_msaamp
HQ=perf/c12_orchestrator/pair_guard/host_quiet.py
gate() { local i; for i in $(seq 1 60); do python3 $HQ --quiet && { echo "host_quiet GREEN before $1 load[$(cut -d' ' -f1-3 /proc/loadavg)] $(date -u +%FT%TZ)" | tee -a perf/of3t_msaamp/HOST_QUIET.txt; return 0; }; sleep 10; done; echo "host_quiet RED before $1"; return 1; }
run() { gate "$1" || exit 1; bash perf/of3t_msaamp/run_ours.sh "$@" 2>&1 | grep -E "^===|FWD|mass-weighted|worst"; }
run ours_legacy --legacy-orientation
run ours_fixed
run ours_fixed_AA
BND=$S/crop64.pt run crop64_ours_fixed
$PY perf/of3t_msaamp/summarize.py "" "" ours_fixed | tail -30
$PY perf/of3t_msaamp/summarize.py crop64_ $S/crop64.pt crop64_ours_fixed | grep -A30 '"mass_weighted"' | head -60
