#!/usr/bin/env bash
# bcx-mutate: the rejection profile on a tree that carries the checkpoint fix.
#
# Every device acceptance reading this campaign holds was produced before 1127f9ea8, where the
# card ran a different checkpoint from the JAX around it on 19 of 20 draws
# (perf/bcx_mutate/ckpt_mismatch_by_tree.py). So the profile -- which stage, which filter, how
# often -- has to be measured again. bcx-multimer's in-flight arm is seed 0, one trajectory;
# this is seed 1 and runs until it is stopped, so the two are independent draws of the same
# program on the same tree.
#
# Rooted in this worktree on purpose: a detached job rooted in another slug's worktree gets its
# files deleted under it when that slug concludes.
set -u

WT=/home/ttuser/.coworker/wt/bcx-mutate
ART=/home/ttuser/bcx_mutate_art
CARD=${CARD:-0}
SEED=${SEED:-1}
TRAJ=${TRAJ:-6}
VENV=/home/ttuser/bcx_e2e_venv/bin/python3

mkdir -p "$ART"
cd "$WT" || exit 1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ART/profile_s${SEED}
LOG=$ART/profile_s${SEED}.log
CLK=$ART/profile_s${SEED}_clock.txt

{
  echo "launched   $STAMP"
  echo "commit     $(git rev-parse HEAD)"
  echo "card       $CARD"
  echo "seed       $SEED"
  echo "traj       $TRAJ"
  echo "loadavg    $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$ART/profile_s${SEED}_stamp.txt"

# AICLK sampled DURING the run, not before: a clock is part of any timing this produces.
nohup setsid bash -c "
  while :; do
    printf '%s ' \"\$(date -u +%H:%M:%SZ)\"
    TT_VISIBLE_DEVICES=$CARD ~/.local/bin/tt-smi -s 2>/dev/null \
      | grep -oE '\"aiclk\"[^,]*' | head -1
    printf ' load=%s\n' \"\$(cut -d' ' -f1 /proc/loadavg)\"
    sleep 60
  done
" < /dev/null >> "$CLK" 2>&1 &
echo "clockwatch pid $!" >> "$ART/profile_s${SEED}_stamp.txt"

nohup setsid env \
  TT_VISIBLE_DEVICES=$CARD \
  TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:bcx-mutate \
  "$VENV" -u perf/bcx_predictor/run_arm.py \
    --arm device --multimer-pool --pool-resident 1 --bucket 32 \
    --trajectories "$TRAJ" --seed "$SEED" \
    --params /home/ttuser/bcx_e2e/af2_params \
    --settings /home/ttuser/bcx_e2e/bc2/examples/pdl1.json \
    --out "$OUT" \
  < /dev/null > "$LOG" 2>&1 &

ARM=$!
echo "arm pid $ARM" >> "$ART/profile_s${SEED}_stamp.txt"
echo "arm pid $ARM  log $LOG  out $OUT"
