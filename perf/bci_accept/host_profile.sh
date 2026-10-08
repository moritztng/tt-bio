# Host profile for BCI device jobs: the two boxes differ only in the tree and the interpreter.
# Sourced by card_ab.sh and card_campaign.sh after CARD is set, so the per-card lock can use it.
# qb1's cards are single-chip p150a and its tree is the integration branch plus this one; qb2's
# are p300 and the tree is this worktree. BCI_WT, BCI_PY and BCI_LOCK override, which is how a
# private clone on either box gets used instead of the default tree.
#
# CLK_NODE is the /sys/class/tenstorrent/tenstorrent!N the AICLK sampler must read, and on qb1 it is
# NOT the card number. state/bci/CHIPS.md: "qb1 logical card N (TT_VISIBLE_DEVICES) is NOT node N:
# 0 = node 1, 1 = node 2, 2 = node 3, 3 = node 0." Sampling tenstorrent!$CARD there reads a
# DIFFERENT board -- on 10-07 that was rel0121's chain A -- and it answers with a perfectly
# plausible 1350, so the reading looks fine and is about someone else's work.
case "${BCI_HOST_NAME:-$(hostname)}" in
  tt-quietbox2) WT=${BCI_WT:-/home/ttuser/.coworker/wt/bci-accept}
                PY=${BCI_PY:-/home/ttuser/fdv_fresh/venv/bin/python}
                LOCK=${BCI_LOCK:-/home/ttuser/bci_chip$CARD.lock}
                CLK_NODE=$CARD ;;
  tt-quietbox)  WT=${BCI_WT:-/home/ttuser/bci_int}
                PY=${BCI_PY:-/home/ttuser/bcx_e2e_venv/bin/python}
                LOCK=${BCI_LOCK:-/home/ttuser/bci_qb1_card$CARD.lock}
                case "$CARD" in
                  0) CLK_NODE=1 ;; 1) CLK_NODE=2 ;; 2) CLK_NODE=3 ;; 3) CLK_NODE=0 ;;
                  *) echo "no qb1 node mapping for logical card $CARD" >&2; exit 1 ;;
                esac ;;
  *) echo "no BCI host profile for ${BCI_HOST_NAME:-$(hostname)}; set BCI_WT, BCI_PY and BCI_LOCK" >&2; exit 1 ;;
esac
TARGET=/home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb
AF2=/home/ttuser/bcx_e2e/af2_params
# How long to wait for the per-card flock. The default is short on purpose -- a device job that
# cannot get the lock should say so rather than hang -- but a job deliberately chained behind
# another BCI job on the same card sets this to the chain's expected length.
LOCK_WAIT=${BCI_LOCK_WAIT:-60}
# An optional suffix on this job's artifacts, for running the SAME card, length and script
# under a different lever -- BCI_TAG=softmax_fp32 beside the untagged control. Empty by
# default, so a plain run keeps the name it had. Without it the second arm of a lever A/B
# overwrites the first and the comparison is gone before anyone reads it.
TAG=${BCI_TAG:+_$BCI_TAG}
