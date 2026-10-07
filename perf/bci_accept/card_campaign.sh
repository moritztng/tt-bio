#!/bin/bash
# The #17 paired comparison, device arm, launched under state/bci/CHIPS.md's RULES.
#
#   bash perf/bci_accept/card_campaign.sh <chip> [trajectories] [binder_length]
#
# Opens a device. Run it only when the NOW block of state/bci/CHIPS.md names bci-accept as the
# holder of <chip>. One BCI device job at a time: the flock below enforces that, so do not bypass
# it. Never `tt-smi -r` from here; board (0,1) is the booth's.
#
# This is the arm Siddhant ran, reproduced deliberately rather than by default: the design loop's
# Evoformer on card, every other stage and the whole validation ensemble on host JAX
# (campaign_predictor's validation="jax"), design_dropout at BindCraft 2's own true, and
# --evoformer-dropout ignore to waive this branch's refusal so the trunk folds dropout-free the way
# his did. Without that waiver the run stops at the first swap, which is the fix working, not a
# reproduction.
#
# --full is not optional here, it is the whole reason this job needs a chip at all. Acceptance only
# exists in a full campaign: a trajectory-only run prints "no design will be accepted" and skips the
# ProteinMPNN redesign and the validation ensemble, so without --full this burns a chip slot and
# still cannot report the 0 of 8 / 3 of 8 the issue turns on. The gradient trajectories are the fast
# part; the redesign and validation that follow them run on host JAX in both arms by design.
#
# Its host control is the same script with --trunk jax at the same campaign seed, which runs
# card-free and is already measured on pc. Both arms pin --trajectories-per-card 1: the interleaving
# width is chosen from free host memory, so unpinned arms on two boxes draw different widths.
set -u
CARD=${1:?chip number from CHIPS.md NOW}
TRAJ=${2:-3}
LEN=${3:-60}
# shellcheck source=perf/bci_accept/host_profile.sh
. "$(dirname "${BASH_SOURCE[0]}")/host_profile.sh"
OUT=$WT/.bci/card_campaign_chip$CARD.log
CLK=$WT/.bci/card_campaign_chip$CARD.aiclk
PROJ=$WT/.bci/proj_card_chip$CARD

cd "$WT" || exit 1
[ -e /home/ttuser/fdv_perf_quiet ] && { echo "fdv_perf_quiet exists: FDV is measuring, not starting" >&2; exit 1; }
# A project folder from an earlier attempt carries .campaign_state.json, and BindCraft 2 resumes
# against it: the run would skip the trajectories it thinks it already attempted and the arms would
# stop being paired by trajectory number.
[ -e "$PROJ" ] && { echo "$PROJ exists; move it aside, a resume is not a rerun" >&2; exit 1; }

export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:$WT
export TT_VISIBLE_DEVICES=$CARD

# The AICLK sets the fold time, so it is sampled DURING the run, not before it. A dead ARC answers
# 4294967295 without raising, so that value is a sentinel, not a reading.
sample_aiclk() {
  while :; do
    printf '%s %s\n' "$(date -u +%H:%M:%SZ)" \
      "$(cat "/sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk" 2>/dev/null || echo unread)"
    sleep 10
  done > "$CLK"
}
sample_aiclk &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

{
  echo "=== card campaign, chip $CARD, $TRAJ trajectories, binder $LEN ==="
  git -C "$WT" rev-parse HEAD
  date -u +'start %Y-%m-%dT%H:%M:%SZ'
  flock -w 60 "$LOCK" \
    nice -n 10 timeout 14400 "$PY" perf/bci_accept/capture_logits.py \
      --trunk card --card "$CARD" --evoformer-dropout ignore \
      --target-pdb "$TARGET" --af2-weights "$AF2" \
      --out "$WT/.bci/logits_card_chip$CARD.npz" \
      --project "$PROJ" --design-dropout true --full \
      --trajectories "$TRAJ" --trajectories-per-card 1 \
      --binder-lengths "$LEN" "$LEN"
  echo "=== rc=$? ==="
  date -u +'end %Y-%m-%dT%H:%M:%SZ'
} > "$OUT" 2>&1

kill $SAMPLER 2>/dev/null
echo "wrote $OUT, AICLK samples in $CLK"
