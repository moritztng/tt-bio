#!/usr/bin/env bash
# bcx-mutate: the rejection profile on a tree that carries the checkpoint fix.
#
# Every device acceptance reading this campaign holds was produced before 1127f9ea8, where the
# card ran a different checkpoint from the JAX around it on 19 of 20 draws
# (perf/bcx_mutate/ckpt_mismatch_by_tree.py). So the profile -- which stage, which filter, how
# often -- has to be measured again.
#
# WT selects the tree the arm executes. cwd wins over the venv: with `cd $WT`, `import tt_bio`
# resolves to $WT/tt_bio and not to the _ttbio.pth entry, verified before the first armtree
# launch. That is the whole reason a second tree can be run without touching this worktree's
# files under a live arm.
#
# TAG separates one tree's artifacts from another's, because a rejection is only readable next
# to the tree that produced it and a shared .campaign_state.json would pool two trees into one
# profile.
set -u

WT=${WT:-/home/ttuser/.coworker/wt/bcx-mutate}
ART=/home/ttuser/bcx_mutate_art
CARD=${CARD:-0}
SEED=${SEED:-1}
TRAJ=${TRAJ:-6}
TAG=${TAG:-}
VENV=/home/ttuser/bcx_e2e_venv/bin/python3

mkdir -p "$ART"
cd "$WT" || exit 1

NAME=profile_s${SEED}${TAG}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ART/$NAME
LOG=$ART/$NAME.log
CLK=$ART/${NAME}_clock.txt

{
  echo "launched   $STAMP"
  echo "tree       $WT"
  echo "commit     $(git rev-parse HEAD)"
  echo "card       $CARD"
  echo "seed       $SEED"
  echo "traj       $TRAJ"
  echo "loadavg    $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$ART/${NAME}_stamp.txt"

# AICLK sampled DURING the run, not before: a clock is part of any timing this produces.
# tt-smi honours TT_VISIBLE_DEVICES, so `head -1` is this card and not card 0 -- checked by
# reading 800 on an idle card 3 while card 0 read 1350 in the same minute.
nohup setsid bash -c "
  while :; do
    printf '%s ' \"\$(date -u +%H:%M:%SZ)\"
    TT_VISIBLE_DEVICES=$CARD ~/.local/bin/tt-smi -s 2>/dev/null \
      | grep -oE '\"aiclk\"[^,]*' | head -1
    printf ' load=%s\n' \"\$(cut -d' ' -f1 /proc/loadavg)\"
    sleep 60
  done
" < /dev/null >> "$CLK" 2>&1 &
echo "clockwatch pid $!" >> "$ART/${NAME}_stamp.txt"

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
echo "arm pid $ARM" >> "$ART/${NAME}_stamp.txt"
echo "arm pid $ARM  log $LOG  out $OUT  tree $WT"
