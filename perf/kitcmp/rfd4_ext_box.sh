#!/bin/bash
# The phases session 4's box_rfd4.sh did not carry, run on that same box after it finishes.
#   bash rfd4_ext_box.sh <climb-deadline> <workers-deadline> <eval-deadline>   (unix epochs)
# Session 4 was launched from 3980af393, before the raised climb ceilings and the worker sweep existed, and a
# running session is never edited. The box is already set up and paid for, so the extension runs here rather
# than in a fifth rental: one box, one protocol, one session.
# Markers under /root/kc/rfd4: EXT-CLIMB-DONE, WORKERS-DONE, RESCORE-DONE, EXT-DONE.
set -u
CLIMB_D=${1:?climb deadline epoch} WORK_D=${2:?workers deadline epoch} EVAL_D=${3:?eval deadline epoch}
B=/root/kc/rfd4; P=/opt/rfd4-venv/bin/python
say(){ echo "$(date -u +%FT%TZ) ext: $*"; }

# Two runs on one card measure neither, so nothing starts until the session's own phases are done.
while [ ! -f $B/RFD4-DONE ] && [ ! -f $B/RFD4-FAILED ]; do
  [ $(date +%s) -ge $CLIMB_D ] && { say "box_rfd4.sh still running at the climb deadline -- extension skipped"; exit 1; }
  sleep 30
done
say "box_rfd4.sh finished; extension starts"

# --- the climb the old ceilings cut short ----------------------------------------------------------------
# Session 4's 128 climb ended ON b128, the configured ceiling, still gaining 4.7 % a rung: that rate was our
# cost guard's, not the card's. --sweep-only re-climbs the top arms under the raised ceilings and reuses the
# rungs this same box already measured, so it pays only for the new ones. Writes ARMS-EXT.json, never over
# ARMS.json.
if [ $(date +%s) -lt $CLIMB_D ]; then
  $P /root/kc/rfd4_arms_w.py --inputs /root/kc/inputs --out $B --deadline $CLIMB_D \
    --sweep-only --arms 2 --screen-warm 5 --final-warm 9 > $B/ext-climb.log 2>&1
  echo $? > $B/EXT-CLIMB-DONE; say "extended climb done"
fi

# --- the host-parallelism knob, at each size's published arm and batch ------------------------------------
# The card was above 50 % util for 35 % of a 128-token b128 batch and drew 195 W of 1000 W, so the rate we
# were about to publish as NVIDIA's was set by host featurization. num_workers is shipped
# (configs/inference.yaml:79) and changes what the host overlaps, never what is computed. D-fairgpu-04.
if [ -f $B/ARMS.json ] && [ $(date +%s) -lt $WORK_D ]; then
  $P /root/kc/rfd4_arms_w.py --inputs /root/kc/inputs --out $B --deadline $WORK_D \
    --workers ${WORKERS:-4,16,48} --screen-warm 5 --final-warm 9 > $B/workers.log 2>&1
  echo $? > $B/WORKERS-DONE; say "worker sweep done"
fi

# --- metrics for the reference designs, over the CIFs that already exist -----------------------------------
if [ -f $B/REFERENCE.json ] && [ $(date +%s) -lt $EVAL_D ]; then
  $P /root/kc/rfd4_rescore.py $EVAL_D > $B/rescore.log 2>&1
  echo $? > $B/RESCORE-DONE; say "reference rescore done"
fi

echo done > $B/EXT-DONE; say "EXT-DONE"
