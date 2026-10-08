#!/bin/bash
# bci-seventeen: the on-card arm at the REPORTER'S OWN sample size and token axis.
#
# issue #17 reports 0 of 8 on card against 3 of 8 host JAX, campaign_seed=42, design_tokens=288.
# bci-accept pinned the mechanism (#21, fixed in c79d1d0e5) at 3 trajectories and 175 tokens.
# This arm answers the observation in the terms it was made in: 8 trajectories, 288 tokens.
#
# 288 tokens = hPDL1 (115 residues) + binder 173. The reporter's own target is a 172-aa protein we
# do not have, so the TOKEN AXIS is reproduced, not his target. That is the axis that selects the
# on-card kernels, which is what #17 is about.
#
# design_dropout=false on BOTH arms: the on-card Evoformer applies no dropout at all
# (bci-accept's finding), so false is what makes the two arms matched rather than what biases them.
#
# Never re-enters a project folder: BindCraft 2 charges a claimed-but-unfinished trajectory against
# the budget and never retries it, so a resume would silently run fewer than 8.
set -u
CARD=${1:?logical card}
CLK_NODE=${2:?clk node for that card}
WAIT_PID=${3:-}

WT=/home/ttuser/.coworker/wt/bci-seventeen
RUN=/home/ttuser/bci-seventeen-run
PY=/home/ttuser/bcx_e2e_venv/bin/python
BC2=/home/ttuser/bcx_e2e/bc2
LOG=$RUN/.bci/card_8traj_288.log
PROJ=$RUN/proj_card_8traj_288

cd "$WT" || exit 1
export PYTHONPATH=$BC2:$WT
export TT_VISIBLE_DEVICES=$CARD
mkdir -p "$RUN/.bci"

exec >> "$LOG" 2>&1
echo "=== bci-seventeen on-card arm: 8 trajectories, 288 tokens, seed 42, BC2 v1.0.1 filters ==="
date -u +"armed %Y-%m-%dT%H:%M:%SZ"
echo "logical card $CARD = /dev/tenstorrent/$CLK_NODE, aiclk tenstorrent!$CLK_NODE"

if [ -n "$WAIT_PID" ]; then
  echo "waiting for the 0.13.1 gate arm pid $WAIT_PID to release the card"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate pid $WAIT_PID gone"
  sleep 60
fi

# A card reads free between two device opens, so confirm it twice a minute apart before taking it.
for probe in 1 2; do
  H=$(fuser "/dev/tenstorrent/$CLK_NODE" 2>/dev/null | tr -d " ")
  if [ -n "$H" ]; then echo "$(date -u +%H:%M:%SZ) probe $probe: node still held by $H, standing down"; exit 3; fi
  [ $probe = 1 ] && sleep 60
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) node /dev/tenstorrent/$CLK_NODE unheld on two probes, taking it"

[ -e "$PROJ" ] && { echo "$PROJ exists; a resume is not a rerun, move it aside"; exit 1; }

CLK=$RUN/.bci/card_8traj_288.aiclk
( while :; do printf "%s %s\n" "$(date -u +%H:%M:%SZ)" \
    "$(cat "/sys/class/tenstorrent/tenstorrent!$CLK_NODE/tt_aiclk" 2>/dev/null || echo unread)"; \
    sleep 30; done > "$CLK" ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

date -u +"start %Y-%m-%dT%H:%M:%SZ"
git -C "$WT" log --oneline -1
flock -w 7200 "/home/ttuser/bci_qb1_card${CARD}.lock" \
  nice -n 10 timeout 144000 "$PY" perf/bci_accept/capture_logits.py \
    --trunk card --card "$CARD" --full \
    --target-pdb "$BC2/settings/target/structures/hPDL1.pdb" \
    --af2-weights /home/ttuser/bcx_e2e/af2_params \
    --out "$RUN/.bci/logits_card_8traj_288.npz" \
    --project "$PROJ" \
    --design-dropout false --trajectories 8 --trajectories-per-card 1 \
    --binder-lengths 173 173
echo "=== rc=$? ==="
date -u +"end %Y-%m-%dT%H:%M:%SZ"
kill $SAMPLER 2>/dev/null
