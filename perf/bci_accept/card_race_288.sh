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
#
# WHY THIS RACES TWO CARDS INSTEAD OF WAITING ON ONE PID (2026-10-08):
# the first version gated on bci-release's parity arm pid 3690907 and an estimate that it would
# free logical card 3 near 13:00Z. Reading the arms' real wrappers instead of estimating:
#   logical 3 (node 0) full_parity_gate  timeout -s INT 36000 from 08:36:05Z -> 18:36Z
#   logical 2 (node 3) release_gate      timeout -s INT 21600 from 09:16:05Z -> 15:16Z
# so the pid being waited on was the LATER of the two by 3h20m, on a row whose own run is 8-14 h.
# Those are caps, not predictions, and either can finish early -- which is the point: this takes
# whichever actually frees first rather than betting on one.
#
# logical card 0 is deliberately NOT a candidate: its ARC is dead (tt_aiclk/tt_arcclk/tt_axiclk all
# read 0xFFFFFFFF, heartbeat frozen), so an arm there could not report the AICLK this row owes
# beside every number. logical card 1 is pfm-acc's, not BCI's to take.
set -u

# "logical:clknode", in the order we would prefer them; the race, not the order, decides.
CANDIDATES=${1:-"2:3 3:0"}

WT=/home/ttuser/.coworker/wt/bci-seventeen
RUN=/home/ttuser/bci-seventeen-run
PY=/home/ttuser/bcx_e2e_venv/bin/python
BC2=/home/ttuser/bcx_e2e/bc2
LOG=$RUN/.bci/card_8traj_288.log
PROJ=$RUN/proj_card_8traj_288
DEAD=4294967295

cd "$WT" || exit 1
export PYTHONPATH=$BC2:$WT
mkdir -p "$RUN/.bci"
exec >> "$LOG" 2>&1

echo "=== bci-seventeen on-card arm: 8 trajectories, 288 tokens, seed 42, BC2 v1.0.1 filters ==="
date -u +"armed %Y-%m-%dT%H:%M:%SZ"
echo "racing for the first free card among: $CANDIDATES (logical:clknode)"

[ -e "$PROJ" ] && { echo "$PROJ exists; a resume is not a rerun, move it aside"; exit 1; }

CARD=""; CLK_NODE=""
while [ -z "$CARD" ]; do
  for cand in $CANDIDATES; do
    c=${cand%%:*}; n=${cand##*:}
    LOCK=/home/ttuser/bci_qb1_card${c}.lock

    # Hold the lock on a fd that outlives this test, so winning the race means KEEPING the card
    # for the whole campaign rather than releasing it between the check and the run.
    exec 9>>"$LOCK" || continue
    flock -n 9 || { exec 9>&-; continue; }

    # Holding the lock is not the same as the card being free: a neighbour that never took the
    # lock still has the device open. A card also reads free between two device opens, so confirm
    # twice a minute apart before committing an 8-14 h campaign to it.
    ok=1
    for probe in 1 2; do
      H=$(fuser "/dev/tenstorrent/$n" 2>/dev/null | tr -d " ")
      if [ -n "$H" ]; then
        echo "$(date -u +%H:%M:%SZ) card $c: hold the lock but node $n still open by $H, releasing"
        ok=0; break
      fi
      [ $probe = 1 ] && sleep 60
    done
    [ $ok = 1 ] || { exec 9>&-; continue; }

    # A dead ARC answers tt_aiclk with 0xFFFFFFFF without raising. A sentinel is not a reading,
    # and this row owes an AICLK beside every number, so refuse the card rather than report it.
    A=$(cat "/sys/class/tenstorrent/tenstorrent!$n/tt_aiclk" 2>/dev/null || echo unread)
    if [ "$A" = "$DEAD" ] || [ "$A" = "unread" ] || [ -z "$A" ]; then
      echo "$(date -u +%H:%M:%SZ) card $c: tt_aiclk reads '$A' (dead-ARC sentinel or unreadable), refusing"
      exec 9>&-; continue
    fi

    CARD=$c; CLK_NODE=$n
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WON logical card $CARD = /dev/tenstorrent/$CLK_NODE, aiclk ${A} kHz/MHz as reported"
    break
  done
  [ -z "$CARD" ] && sleep 60
done

export TT_VISIBLE_DEVICES=$CARD

CLK=$RUN/.bci/card_8traj_288.aiclk
( while :; do printf "%s %s\n" "$(date -u +%H:%M:%SZ)" \
    "$(cat "/sys/class/tenstorrent/tenstorrent!$CLK_NODE/tt_aiclk" 2>/dev/null || echo unread)"; \
    sleep 30; done > "$CLK" ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

date -u +"start %Y-%m-%dT%H:%M:%SZ"
git -C "$WT" log --oneline -1
# No second flock here: fd 9 already holds this card's lock for the life of this script.
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
