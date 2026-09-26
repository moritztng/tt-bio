#!/usr/bin/env bash
# The matched A/B for this row. ONE commit, ONE card, back to back, same argv.
#
# `arm old` checks the pre-row `optim.py` out of 42a664fa6^ into the tree, runs, and puts the
# current one back. That is the only file the row touched in `tt_bio/`, so the two arms differ
# by exactly the lever and by nothing else -- not by a tree, not by a board, and not by the
# host's memory, which §2 of the state doc measures at 3-4x on this same work.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=${1:?arm: old|new}
OUT=perf/of3t_p10host/out/step_${ARM^^}_384
ARGS="--tokens 384 --cycles 4 --samples 4 --reps 3"
if [ "$ARM" = old ]; then
  cp tt_bio/train/optim.py /tmp/optim_new_$$.py
  trap 'cp /tmp/optim_new_'"$$"'.py tt_bio/train/optim.py; rm -f /tmp/optim_new_'"$$"'.py' EXIT
  git show 42a664fa6^:tt_bio/train/optim.py > tt_bio/train/optim.py
fi
echo "loadavg $(cut -d' ' -f1-3 /proc/loadavg)  MemAvailable $(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-p10host \
  /home/moritz/tt-bio/env/bin/python perf/of3t_p10host/armrun.py "exact_off_$ARM" "$OUT.json" \
  $ARGS > "$OUT.log" 2>&1
echo "rc=$? -> $OUT.json"
