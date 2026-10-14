#!/bin/bash
# gapboard_chain.sh CARD SHA9 (copied to qb1 as ~/spd/gapboard/gb_chain.sh) (on qb1, launched by scripts/gapboard.py): the p150a GAP column for main SHA9 on ONE card.
# Holds the card's flock for the whole column (one tree, one card, one clock regime), hands it back to a release-gate
# leg between cells, records per cell: run.log (start/end rc, node), aiclk.tsv (AICLK of every node + loadavg every
# 0.5 s DURING the cell), harness.log, result.json. Jobs: ~/spd/gapboard/out/SHA9/jobs.txt, "NAME TIMEOUT CMD...".
set -u
CHIP=$1 SHA=$2; G=~/spd/monthpost; T=$G/trees/m$SHA; O=$G/out/$SHA
mkdir -p $G/leases
export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-monthpost TT_BIO_LEASE_DIR=$G/leases
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/ | head -1); export PATH=${VENV%/}/bin:$PATH
# UMD logical ids follow PCI order, /dev nodes do not (qb1: logical 1 = node 2); aiclk.tsv column 2 is the NODE
NODE=$(for n in /sys/class/tenstorrent/tenstorrent\!*; do echo "$(basename "$(readlink -f "$n/device")") ${n##*!}"; done | sort | sed -n "$((CHIP + 1))p" | cut -d" " -f2)
REL="(spd_qb1_card|TT_VISIBLE_DEVICES=)${CHIP}[. ].*(release_gate|gate_fanout|perf_ab)"
# The scoreboard outranks experiments: while this chain waits for or holds the card, a sentinel whose argv matches REL
# makes every qb1 runner that yields to a release-gate leg (the 22 row scripts, feed waiters) hand the card over too.
# 10-13: a plain flock race lost card 0 to a 1 h bench that queued 50 s after the chain, leaving GAP flat for hours.
:
rel() { pgrep -f "$REL" >/dev/null; }
take() { while :; do while rel; do sleep 60; done
  exec 9> "/home/ttuser/spd_qb1_card$CHIP.lock"; flock -w 43200 9 || return 1
  rel || return 0; exec 9>&-; sleep 60; done; }
echo "$(date -u +%FT%TZ) chain start card $CHIP node $NODE pid $$" >> $O/chain.log
take || { echo "$(date -u +%FT%TZ) flock timeout" >> $O/chain.log; exit 3; }
cd $T
while read -r N TM CMD; do
  [ -z "$N" ] || [ "${N:0:1}" = "#" ] && continue
  [ -f $O/$N/run.log ] && grep -q "end rc=0" $O/$N/run.log && continue
  rel && { exec 9>&-; take || exit 3; }
  mkdir -p $O/$N; say() { echo "$(date -u +%FT%TZ) $*" >> "$O/$N/run.log"; }
  say "start chip $CHIP (node $NODE) tree $SHA venv $VENV host $(hostname) cmd: $CMD"
  ( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done ) 9>&- >> "$O/$N/aiclk.tsv" 2>/dev/null &
  S=$!
  timeout -s TERM $((TM + 120)) timeout -s INT "$TM" bash -c "$CMD" > "$O/$N/harness.log" 2>&1 9>&-
  rc=$?; kill $S 2>/dev/null; say "end rc=$rc"
done < $O/jobs.txt
echo "$(date -u +%FT%TZ) chain done" >> $O/chain.log
