#!/usr/bin/env bash
# What the landed AdamW change does to the step's PEAK HOST RESIDENT SET, and to the host
# arithmetic that peak buys back. One commit, one card, back to back, same argv.
#
# `old` checks the pre-row `tt_bio/train/optim.py` out of 42a664fa6^ and puts it back. That is
# the only engine file this row touched, so the arms differ by the lever and nothing else.
# Both arms go through `armrun.py`, which is the exactness-OFF wrapper -- `fullstep.py` run
# directly is an exactness-ON step and costs ~25x.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=${1:?arm: old|new}; shift
OUT=perf/of3t_p10host/out/peak_${ARM}
if [ "$ARM" = old ]; then
  cp tt_bio/train/optim.py "/tmp/optim_new_$$.py"
  trap 'cp "/tmp/optim_new_'"$$"'.py" tt_bio/train/optim.py; rm -f "/tmp/optim_new_'"$$"'.py"' EXIT
  git show 42a664fa6^:tt_bio/train/optim.py > tt_bio/train/optim.py
fi
echo "[$ARM] loadavg $(cut -d' ' -f1-3 /proc/loadavg)  MemAvailable $(awk '/MemAvailable/{printf "%.2f GiB", $2/1048576}' /proc/meminfo)"
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-p10host \
  timeout 900 /home/moritz/tt-bio/env/bin/python perf/of3t_p10host/armrun.py "peak_$ARM" "$OUT.json" \
  --tokens 384 --cycles 4 --samples 4 --reps 3 "$@" > "$OUT.log" 2>&1
echo "[$ARM] rc=$? -> $OUT.json"
grep -E "^PEAK RSS" "$OUT.log" || true
