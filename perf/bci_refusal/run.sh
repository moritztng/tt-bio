#!/bin/bash
# Card legs for #19/#20 on qb2 chip 1, once state/bci/CHIPS.md grants it to bci-refusal.
#   perf/bci_refusal/run.sh <out-tag> [wait]
# With `wait` it queues on the chip lock instead of giving up, then holds the lock while FDV's
# perf window quiet-gates qb2.
# The fixed tree (this worktree) runs success, refusal and sigterm; the unfixed tree (f25be1624,
# extracted with git archive) runs success and refusal as the control.
set -uo pipefail
cd "$(dirname "$0")/../.."
tree=$PWD; tag=${1:?tag}; mode=${2:-now}; out=$tree/perf/bci_refusal/out/$tag
before=$HOME/scratch/bci-refusal-before
mkdir -p "$out"
[ -e /home/ttuser/fdv_perf_quiet ] && { echo "FDV perf rerun is quiet-gating qb2; not starting"; exit 3; }
export TT_VISIBLE_DEVICES=1 BCX_BC2=/home/ttuser/bcx_e2e/bc2
PY=/home/ttuser/bcx_e2e_venv/bin/python3
{
  echo "start=$(date -u +%FT%TZ) head=$(git rev-parse --short HEAD) card=$TT_VISIBLE_DEVICES"
  exec 9>/home/ttuser/bci_chip1.lock
  if [ "$mode" = wait ]; then
    flock 9
    while [ -e /home/ttuser/fdv_perf_quiet ] || pgrep -f perf_ab_r2 >/dev/null; do sleep 60; done
  else
    flock -n 9 || { echo "chip 1 lock held"; exit 4; }
  fi
  echo "locked=$(date -u +%FT%TZ)"
  nice -n 10 timeout -s INT 5400 $PY perf/bci_refusal/driver.py --tree "$tree" --out "$out/fixed" \
      --py $PY --legs success,refusal,sigterm
  echo "fixed_rc=$?"
  [ -d "$before/tt_bio" ] && nice -n 10 timeout -s INT 3600 $PY perf/bci_refusal/driver.py \
      --tree "$before" --out "$out/before" --py $PY --legs success,refusal
  echo "before_rc=$? end=$(date -u +%FT%TZ)"
} > "$out/run.log" 2>&1
