#!/usr/bin/env bash
# of3t-msaamp: every arm, each started only once host_quiet.py is green (loadavg <= 2.0, no other
# device holder making progress, no release gate). The arms' own CPU load decays between them.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-msaamp
cd "$W" || exit 1
HQ=perf/c12_orchestrator/pair_guard/host_quiet.py
gate() {
  local i
  for i in $(seq 1 60); do python3 $HQ --quiet && { echo "host_quiet GREEN before $1 load[$(cut -d' ' -f1-3 /proc/loadavg)] $(date -u +%FT%TZ)" | tee -a perf/of3t_msaamp/HOST_QUIET.txt; return 0; }; sleep 10; done
  echo "host_quiet RED before $1 after 10 min: $(python3 $HQ 2>&1 | tail -3)" | tee -a perf/of3t_msaamp/HOST_QUIET.txt; return 1
}
for spec in "D ours" "D ours_AA" "U bf16auto bf16auto" "U bf16auto bf16auto_AA" "U f32 f32" "U f64 f64" \
            "U bf16auto bf16in --bf16-inputs" "U bf16auto bf16auto_BREAK --break-cot"; do
  set -- $spec; kind=$1; shift
  gate "$*" || exit 1
  if [ "$kind" = D ]; then bash perf/of3t_msaamp/run_ours.sh "$@" 2>&1 | grep -E "^===|FWD|mass-weighted"
  else bash perf/of3t_msaamp/run_arm.sh "$@" 2>&1 | grep -E "^===|HARD|real-block|backward [0-9]"; fi
done
echo SUITE_DONE
