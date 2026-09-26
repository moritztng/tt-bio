#!/bin/bash
# usage: rerecord.sh <name> [tap_gate args...] -- af2ig-trunk-device arm from this worktree on card 3,
# host arms shared through perf/bcx_land/tap_gate_cached.py, sysfs AICLK + loadavg sampled throughout.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
wt=/home/ttuser/.coworker/wt/bcx-land
out=$wt/perf/bcx_land/rerecord; mkdir -p $out
name=$1; shift
( while true; do echo "$(date -u +%H:%M:%S) aiclk=$(aiclk 0) load=$(cut -d' ' -f1 /proc/loadavg)"; sleep 10; done ) > $out/$name.clock &
s=$!
cd $wt
start=$(date -u +%s)
ENV_CACHE=/home/ttuser/scratch/bcx-land-envcache PYTHONPATH=$wt TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-land \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/bcx_land/tap_gate_cached.py \
  --params /home/ttuser/.boltz/af2/params/params_model_1_ptm.npz --stage complex --device "$@" \
  > $out/$name.stdout 2> $out/$name.stderr
echo "rc=$? wall=$(( $(date -u +%s) - start ))s commit=$(git rev-parse HEAD)" > $out/$name.rc
kill $s
