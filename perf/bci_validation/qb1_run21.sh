#!/usr/bin/env bash
# Issue #21 on qb1 card 2 (logical, = /dev/tenstorrent/3): the fixed arm end to end, 8 trajectories,
# on wk/bci-integration (the fix + the #18 memory fix, so the leak cannot end the run at traj 4).
set -u
TREE=/home/ttuser/bci_int
OUT=/home/ttuser/bci_card21_qb1/fixed
LOG=/home/ttuser/bci_card21_qb1/run.log
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=2
export PYTHONPATH=$TREE:$TREE/perf/bcx_stack:$TREE/perf/bcx_round:$TREE/perf/bcx_predictor
cd "$TREE" || exit 1
{
  echo "=== qb1 card 2 (node 3) #21 fixed arm, started $(date -u +%FT%TZ)"
  git -C "$TREE" rev-parse HEAD
  "$HOME/bcx_e2e_venv/bin/python" -c "import ttnn, jax; print('jax', jax.__version__)"
} >>"$LOG" 2>&1
flock -w 60 /home/ttuser/bci_qb1_card2.lock \
  nice -n 10 timeout -s INT -k 300 7h \
  "$HOME/bcx_e2e_venv/bin/python" "$TREE/perf/bci_validation/card_campaign.py" \
    --arm fixed --ttbio "$TREE" --out "$OUT" \
    --binder 90 --designs 10 --max-trajectories 8 --seed 100 --resident 1 \
    >>"$LOG" 2>&1
echo "=== rc=$? finished $(date -u +%FT%TZ)" >>"$LOG"
