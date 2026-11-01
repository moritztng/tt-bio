#!/bin/bash
# Fix B A/B arm (D-orch-25) on qb1: the partial grade's 33 cases folded with tree SHA (wk/mol1-wh1024-prof, Mol-1
# samples MSA rows per cycle) on one card, seed-major so every seed pass is a gradeable A/B on all 33 cases.
#   bash qb1_fixb.sh CARD [SECONDS]      log ~/mol1-quality/fixb/run-c<CARD>.log, FIXB-EXIT at the end
# Several cards may run it at once: run_tt.sh claim files share ~/mol1-quality/fixb/tt-bh-b.
set -uo pipefail
cd ~/mol1-quality; C=${1:?card}; S=${2:-43200}; O=mol1-quality-fixb-c$C; SHA=246a0237c
say() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p fixb
if [ ! -f fixb/tree/.ok ]; then
  rm -rf fixb/tree && mkdir fixb/tree && git -C ~/tt-bio-dev archive $SHA | tar -x -C fixb/tree && echo $SHA > fixb/tree/.ok \
    || { say "archive failed"; exit 1; }
fi
export PATH=$HOME/tt-bio-dev/env/bin:$PATH PYTHONPATH=$HOME/mol1-quality/fixb/tree LOCK_HOST=qb1
say "engine $(python3 -c 'import tt_bio; print(tt_bio.__file__)') = $SHA"
until [ "$(python3 ~/spd/feed/book.py active "$C" 2>/dev/null)" = "$O" ]; do
  say "book $(python3 ~/spd/feed/book.py add $O "$C" "$S" --note "mol1-quality Fix B A/B ($SHA)")"; sleep 60; done
flock "$HOME/spd_qb1_card$C.lock" true  # the outgoing holder finishes its fold first
say "chip $C: window open"
for s in 42 66 101 2024 8888; do
  bash code/run_tt.sh --book $O --cases fixb/CASES.json inputs fixb/tt-bh-b "$s" "$C"
  [ "$(python3 ~/spd/feed/book.py active "$C" 2>/dev/null)" = "$O" ] || { say "window over at seed $s"; break; }
done
python3 ~/spd/feed/book.py rm $O; say "FIXB-EXIT"
