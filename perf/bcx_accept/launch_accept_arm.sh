#!/bin/bash
# bcx-accept: one detached acceptance arm on qb2 card 1.
#
# The output dir is removed first, deliberately. BindCraft 2 loads .campaign_state.json from an
# existing dir and keeps its "attempted"/"trajectories" fields, so a reused dir reports
# trajectories another program ran: profile_s1 counts pdl1_denovo_l111_e04191bcdbcdfb6c, whose
# directory is empty and predates that arm by 11 minutes. !_Trajectories.csv is the ledger that
# only records what this arm finished; .campaign_state.json is not.
set -u
ART=/home/ttuser/bcx_accept_art
TREE=$ART/armtree
SEED=${SEED:-3}
CARD=${CARD:-1}
TRAJ=${TRAJ:-6}
OUT=$ART/accept_s$SEED
PY=/home/ttuser/bcx_e2e_venv/bin/python3

cd "$TREE" || exit 1
rm -rf "$OUT"

{
  echo "launched   $(date -u +%Y%m%dT%H%M%SZ)"
  echo "tree       $TREE"
  echo "commit     $(git rev-parse HEAD)"
  echo "dirty      $(git status --porcelain | wc -l) files"
  echo "ckpt_fix   $(git merge-base --is-ancestor 1127f9ea8 HEAD && echo present || echo ABSENT)"
  echo "weakref    $(grep -c weakref.ref tt_bio/autograd.py)"
  echo "card       $CARD"
  echo "seed       $SEED"
  echo "traj       $TRAJ"
  echo "loadavg    $(cut -d" " -f1-3 /proc/loadavg)"
} > "$ART/accept_s${SEED}_stamp.txt"

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-accept
export OMP_NUM_THREADS=6

$PY -u perf/bcx_predictor/run_arm.py --arm device --multimer-pool --pool-resident 1 \
    --bucket 32 --trajectories "$TRAJ" --seed "$SEED" \
    --params /home/ttuser/bcx_e2e/af2_params \
    --settings /home/ttuser/bcx_e2e/bc2/examples/pdl1.json \
    --out "$OUT" > "$ART/accept_s$SEED.log" 2>&1 < /dev/null &
PID=$!
echo $PID > "$ART/accept_s$SEED.pid"
echo "arm pid $PID" >> "$ART/accept_s${SEED}_stamp.txt"

# AICLK sampled DURING the fold, not before: a perf number without a clock is not a measurement.
# The sampler runs as its own process so the exit status below is the ARM's. It used to be the
# loop condition, and `echo "arm exit $?"` after a `while kill -0` loop reports the status of the
# failing `kill -0`, which is always 0. accept_s3 crashed in binder optimization with a KeyError
# and its stamp read `arm exit 0`. Read the log tail to decide whether a trajectory completed.
(
  while kill -0 $PID 2>/dev/null; do
    CLK=$(TT_VISIBLE_DEVICES=$CARD /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
          | grep -m1 "\"AICLK\"" | grep -oE "0x[0-9a-fA-F]+")
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$((CLK)),$(cut -d" " -f1 /proc/loadavg)" \
         >> "$ART/accept_s${SEED}_clock.csv"
    sleep 120
  done
) &
CLOCK=$!
wait $PID
RC=$?
wait $CLOCK 2>/dev/null
echo "arm exit $RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ART/accept_s${SEED}_stamp.txt"
