#!/bin/bash
# bcx-accept: BindCraft 2's own JAX reference campaign, detached, on qb2's CPU.
#
# The campaign's only reference run was OOM-killed on pc at 2026-09-25T12:54:16Z (pid 3546462,
# anon-rss 15.09 GB, global OOM, dmesg `Out of memory: Killed process 3546462 (python)`). It had
# banked one completed trajectory, 1 accepted, and died in trajectory 2's anneal. So the GO
# condition's reference side stands at n=1 with no seed variance, and pc cannot hold a second one:
# 30 GB of RAM against a 15 GB working set. qb2 has 249 GB and 16 cores.
#
# Two deliberate choices:
#
# * SEED 1, not 0. Repeating seed 0 redraws the trajectory pc already finished. A second seed is
#   what "within seed variance of the shipped example's own count" actually needs.
# * `--bucket 0`, which appends no override at all, so the settings are byte-for-byte the unedited
#   examples/pdl1.json that pc ran (its stamp records `length_bucket_flag: 0`). Passing `--bucket
#   32` would insert a `length_bucket_size` key, and since `design_hash` hashes the settings text
#   and `length_bucket_size` is not in EXCLUDED_SETTING_NAMES, it moves the draw without moving
#   the fold.
#
# Half the cores, by `taskset`. A device arm is normally live on this box and BindCraft 2 spends
# most of a gradient step on the host, so an unpinned 16-thread JAX job would distort another
# row's step times. The cost is our own wall clock, which is not a number this arm is for.
set -u
ART=/home/ttuser/bcx_accept_art
TREE=$ART/armtree
SEED=${SEED:-1}
TRAJ=${TRAJ:-10}
CORES=${CORES:-8-15}
OUT=$ART/ref_s$SEED
PY=/home/ttuser/bcx_e2e_venv/bin/python3
STAMP=$ART/ref_s${SEED}_stamp.txt

cd "$TREE" || exit 1
rm -rf "$OUT"

{
  echo "launched   $(date -u +%Y%m%dT%H%M%SZ)"
  echo "host       $(hostname)"
  echo "tree       $TREE"
  echo "commit     $(git rev-parse HEAD)"
  echo "dirty      $(git status --porcelain | wc -l) files"
  echo "arm        reference (BindCraft 2's AlphaFoldDesignModel, no card, no levers)"
  echo "settings   /home/ttuser/bcx_e2e/bc2/examples/pdl1.json md5 $(md5sum /home/ttuser/bcx_e2e/bc2/examples/pdl1.json | cut -d' ' -f1)"
  echo "seed       $SEED"
  echo "traj       $TRAJ"
  echo "cores      $CORES"
  echo "memfree    $(free -g | awk '/^Mem:/{print $7}') GB available"
  echo "loadavg    $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$STAMP"

export OMP_NUM_THREADS=8

taskset -c "$CORES" $PY -u perf/bcx_predictor/run_arm.py --arm reference --multimer-pool \
    --bucket 0 --trajectories "$TRAJ" --seed "$SEED" \
    --params /home/ttuser/bcx_e2e/af2_params \
    --settings /home/ttuser/bcx_e2e/bc2/examples/pdl1.json \
    --out "$OUT" > "$ART/ref_s$SEED.log" 2>&1 < /dev/null &
PID=$!
echo $PID > "$ART/ref_s$SEED.pid"
echo "arm pid $PID" >> "$STAMP"

# `wait`, not a `kill -0` poll. A polling loop reports the exit status of its own `kill -0`, which
# is what wrote `arm exit 0` under accept_s3's validation crash and made a dead trajectory read as
# a clean one. Read the log tail, never a stamp, to decide whether a trajectory completed.
wait $PID
RC=$?
echo "arm exit $RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STAMP"
