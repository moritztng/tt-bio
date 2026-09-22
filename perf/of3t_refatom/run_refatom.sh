#!/bin/bash
# of3t-refatom deliverable 2: the one device arm, on card 0, 20 rungs.
#
# `of3t-trajwide`'s harness for one arm, with two deliberate differences:
#
#  * TT_VISIBLE_DEVICES is exported for the CLOCK SAMPLER too, so `tt-smi -s` reports only
#    this card. run_arms.sh's `head -4` truncated a four-device dump to device 0 whichever
#    card the arm ran on; here device 0 IS the card, so the log widen_score.py parses is a
#    reading of the chip that did the work, not of its board partner.
#  * one arm, one pass, no lock dance: the reference side is already banked at all 20 rungs
#    (`w/theirs/k20.npz`, 761 tensors including all eight ref_atom_feature_embedder entries).
#
# Resumable: trajwide.py saves per-rung state, so a relaunch continues where it stopped.
set -u
cd "$(dirname "$0")/../.."
CARD=${1:-0}
ARM=refatom
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_runs/trajwide
mkdir -p "$R"
C="$R/chain_${ARM}.log"
say() { echo "$(date -u +%FT%TZ) [card $CARD] $*" >> "$C"; }

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-refatom
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

clock_watch() {
  while :; do
    printf "%s " "$(date -u +%FT%TZ)"
    ~/.local/bin/tt-smi -s 2>/dev/null | tr -d " " | grep -i aiclk | head -4 | tr "\n" " "
    printf "\n"; sleep 60
  done
}

exec 9>"$R/$ARM.lock"
if ! flock -n 9; then say "already running, exiting"; exit 0; fi
say "start pid=$$ cwd=$PWD"
echo "=== $ARM start $(date -u +%FT%TZ) card $CARD" >> "$R/ours_$ARM.log"
clock_watch >> "$R/aiclk_$ARM.log" 2>&1 &
CW=$!
$PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --refatom device --threads 3 \
    >> "$R/ours_$ARM.log" 2>> "$R/ours_$ARM.err"
RC=$?
kill "$CW" 2>/dev/null
echo "=== $ARM done rc=$RC $(date -u +%FT%TZ)" >> "$R/ours_$ARM.log"
echo "arm=$ARM rc=$RC card=$CARD at=$(date -u +%FT%TZ)" > "$R/$ARM.done"
say "done rc=$RC"
exec 9>&-
