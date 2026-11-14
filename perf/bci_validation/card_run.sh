#!/bin/bash
# The issue #21 card leg: one real campaign per arm on the chip BCI has been granted.
#
#   card_run.sh <arm> <card>            arm = fixed | prefix | control
#
# Written to be launched detached and read back later:
#   setsid nohup perf/bci_validation/card_run.sh fixed 1 >/home/ttuser/bci_card_fixed.log 2>&1 &
#
# state/bci/CHIPS.md RULES, all of them: the card comes from the grant line and nowhere else,
# `nice -n 10`, one BCI device job at a time behind the flock, a `timeout` on the run, the mesh
# descriptor set inside the script, AICLK sampled during the run (card_campaign.py does it), and
# no `tt-smi -r` anywhere. A hung run is stopped with SIGINT and then SIGTERM, which is what
# `timeout -s INT -k` sends.
set -euo pipefail
cd "$(dirname "$0")/../.."

arm=${1:?arm: fixed | prefix | control}
card=${2:?the card the grant in state/bci/CHIPS.md names}
hours=${BCI_HOURS:-6}

case "$arm" in
  fixed)   tree=$PWD;                  extra=() ;;
  prefix)  tree=/home/ttuser/bci_prefix; extra=() ;;
  control) tree=/home/ttuser/bci_prefix; extra=(--no-extra-msa) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

out=/home/ttuser/bci_card/$arm
rm -rf "$out"; mkdir -p "$out"

export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export PYTHONPATH=$tree
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:bci-validation
export JAX_COMPILATION_CACHE_DIR=/home/ttuser/bci_card/xlacache

exec flock /home/ttuser/bci_chip"$card".lock \
  nice -n 10 timeout -s INT -k 120 "${hours}h" \
  /home/ttuser/fdv_fresh/venv/bin/python3 -u perf/bci_validation/card_campaign.py \
    --arm "$arm" --ttbio "$tree" --out "$out" "${extra[@]}" "${@:3}"
