#!/bin/bash
# One shipped-example trajectory, past the MPNN validation ensemble, on the routed tree.
#
# Seed 3 is deliberate: bcx-accept's arm at that seed passed all five design stages on its first
# trajectory (i_pTM 0.82 -> 0.85, pLDDT 0.93 -> 0.95), produced its 10 MPNN redesigns and died in
# predict_validation_ensemble. It is the trajectory we know reaches the failing call, so it is
# the one that proves the route open. The mask fix does not move it: its draw is first in its
# 32-bucket, which is the case the stale key served correctly.
#
# The exit status is the ARM's. `echo "arm exit $?"` after a `while kill -0 $PID` loop reports the
# status of the failing kill, which is always 0 -- that is what stamped `arm exit 0` over
# accept_s3's crash and made a dead trajectory read as a completed one. So the clock sampler runs
# as its own process and the status comes from `wait $PID`.
set -u
CARD=${CARD:-1}
SEED=${SEED:-3}
TRAJ=${TRAJ:-1}
TREE=${TREE:-/home/ttuser/bcx_mutate_art/routetree}
ART=${ART:-/home/ttuser/bcx_mutate_art}
PY=/home/ttuser/bcx_e2e_venv/bin/python3
TAG="route_s$SEED"
OUT="$ART/${TAG}_arm"
STAMP="$ART/${TAG}_stamp.txt"
mkdir -p "$ART"

cd "$TREE" || exit 1
{
  echo "launched   $(date -u +%Y%m%dT%H%M%SZ)"
  echo "tree       $TREE"
  echo "commit     $(git rev-parse HEAD)"
  echo "dirty      $(git status --porcelain | wc -l) files"
  echo "card       $CARD"
  echo "seed       $SEED"
  echo "traj       $TRAJ"
  echo "loadavg    $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$STAMP"

TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-mutate \
  $PY -u perf/bcx_predictor/run_arm.py --arm device --multimer-pool --pool-resident 1 \
    --bucket 32 --trajectories "$TRAJ" --seed "$SEED" \
    --params /home/ttuser/bcx_e2e/af2_params \
    --settings /home/ttuser/bcx_e2e/bc2/examples/pdl1.json \
    --out "$OUT" > "$ART/$TAG.log" 2>&1 < /dev/null &
PID=$!
echo $PID > "$ART/$TAG.pid"
echo "arm pid $PID" >> "$STAMP"

# AICLK sampled DURING the run, in its own process so the status above stays the arm's.
(
  while kill -0 $PID 2>/dev/null; do
    CLK=$(TT_VISIBLE_DEVICES=$CARD /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
          | grep -m1 '"aiclk"' | grep -oE '[0-9]+')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${CLK:-},$(cut -d' ' -f1 /proc/loadavg)" \
         >> "$ART/${TAG}_clock.csv"
    sleep 120
  done
) &
CLOCK=$!
wait $PID
RC=$?
wait $CLOCK 2>/dev/null
echo "arm exit $RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STAMP"
