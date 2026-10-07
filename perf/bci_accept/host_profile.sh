# Host profile for BCI device jobs: the two boxes differ only in the tree and the interpreter.
# Sourced by card_ab.sh and card_campaign.sh after CARD is set, so the per-card lock can use it.
# qb1's cards are single-chip p150a and its tree is the integration branch plus this one; qb2's
# are p300 and the tree is this worktree. BCI_WT, BCI_PY and BCI_LOCK override, which is how a
# private clone on either box gets used instead of the default tree.
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
