#!/bin/bash
# The #17 harden A/B on a card, launched under state/bci/CHIPS.md's RULES.
#
#   bash perf/bci_accept/card_ab.sh <chip> <harden-states.pkl> [binder_length]
#
# Opens a device. Run it only when the NOW block of state/bci/CHIPS.md names bci-accept as the
# holder of <chip>. One BCI device job at a time: the flock below is what enforces that, so do not
# bypass it. Never `tt-smi -r` from here; board (0,1) is the booth's.
set -u
CARD=${1:?chip number from CHIPS.md NOW}
STATES=${2:?harden-entry ProteinStates pickle from a host-JAX trajectory}
LEN=${3:-60}
# Two boxes run this and they differ only in the tree and the interpreter, so the script picks
# its own host profile rather than make every caller pass four paths. qb1's cards are single-chip
# p150a and its tree is the integration branch; qb2's are p300 and the tree is this worktree.
case "$(hostname)" in
  tt-quietbox2) WT=${BCI_WT:-/home/ttuser/.coworker/wt/bci-accept}
                PY=${BCI_PY:-/home/ttuser/fdv_fresh/venv/bin/python}
                LOCK=${BCI_LOCK:-/home/ttuser/bci_chip$CARD.lock} ;;
  tt-quietbox)  WT=${BCI_WT:-/home/ttuser/bci_int}
                PY=${BCI_PY:-/home/ttuser/bcx_e2e_venv/bin/python}
                LOCK=${BCI_LOCK:-/home/ttuser/bci_qb1_card$CARD.lock} ;;
  *) echo "no BCI host profile for $(hostname); set BCI_WT, BCI_PY and BCI_LOCK" >&2; exit 1 ;;
esac
TARGET=/home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb
AF2=/home/ttuser/bcx_e2e/af2_params
OUT=$WT/.bci/card_ab_chip$CARD.log
CLK=$WT/.bci/card_ab_chip$CARD.aiclk

cd "$WT" || exit 1
[ -f "$STATES" ] || { echo "no states pickle at $STATES" >&2; exit 1; }
# FDV's perf reruns must not share the box with a device job of ours.
[ -e /home/ttuser/fdv_perf_quiet ] && { echo "fdv_perf_quiet exists: FDV is measuring, not starting" >&2; exit 1; }

export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:$WT
export TT_VISIBLE_DEVICES=$CARD

# The AICLK sets the fold time, so it is sampled DURING the run, not before it: a sample taken while
# the chip is idle says nothing about the clock the work actually ran at. A dead ARC answers
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
  echo "=== card A/B, chip $CARD, binder $LEN, states $STATES ==="
  date -u +'start %Y-%m-%dT%H:%M:%SZ'
  flock -w 60 "$LOCK" \
    nice -n 10 timeout 5400 "$PY" perf/bci_accept/harden_forward_ab.py \
      --states "$STATES" --af2-weights "$AF2" --target-pdb "$TARGET" \
      --binder-length "$LEN" --card "$CARD"
  echo "=== rc=$? ==="
  date -u +'end %Y-%m-%dT%H:%M:%SZ'
} > "$OUT" 2>&1

kill $SAMPLER 2>/dev/null
echo "wrote $OUT, AICLK samples in $CLK"
