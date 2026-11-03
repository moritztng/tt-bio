#!/bin/bash
# B arm of a BH stack A/B (D-orch-25 Fix B, D-orch-41/97 final main) on qb1 or qb2: the 33 cases of fixb/CASES.json
# folded with tree SHA on one card, seed-major so every seed pass is a gradeable A/B on all 33 cases.
#   bash ab_arm.sh SHA OUT CARD [SECONDS]      log on stdout, AB-EXIT at the end
# Several cards of one host may run it at once on the same OUT: run_tt.sh claim files share ~/mol1-quality/OUT.
set -uo pipefail
cd ~/mol1-quality; SHA=${1:?sha}; OUT=${2:?out}; C=${3:?card}; S=${4:-43200}; O=mol1-quality-ab-c$C
say() { echo "$(date -u +%FT%TZ) $*"; }
case $(hostname) in tt-quietbox2) LOCK_HOST=qb2 L=~/spd_qb2_chip$C.lock ;; tt-quietbox) LOCK_HOST=qb1 L=~/spd_qb1_card$C.lock ;;
  *) say "unknown host"; exit 2 ;; esac
P=$(dirname "$(realpath "$0")"); REPO=tt-bio.git; [ -d $REPO ] || REPO=~/tt-bio-dev
if [ ! -f main-$SHA/.ok ]; then
  rm -rf main-$SHA && mkdir main-$SHA && git -C $REPO archive $SHA tt_bio | tar -x -C main-$SHA && echo $SHA > main-$SHA/.ok ||
    { say "archive $SHA failed"; exit 1; }
fi
export PATH=$HOME/tt-bio-dev/env/bin:$HOME/mol1-quality/env/kalign/bin:$PATH PYTHONPATH=$PWD/main-$SHA LOCK_HOST
say "engine $(python3 -c 'import tt_bio; print(tt_bio.__file__)') = $SHA"
until [ "$(python3 ~/spd/feed/book.py active "$C" 2>/dev/null)" = "$O" ]; do
  say "book $(python3 ~/spd/feed/book.py add $O "$C" "$S" --note "mol1-quality BH stack A/B ($SHA)")"; sleep 60; done
flock "$L" true  # the outgoing holder finishes its fold first
say "chip $C: window open"
for s in 42 66 101 2024 8888; do
  bash "$P/run_tt.sh" --book $O --cases fixb/CASES.json inputs "$OUT" "$s" "$C"
  [ "$(python3 ~/spd/feed/book.py active "$C" 2>/dev/null)" = "$O" ] || { say "window over at seed $s"; break; }
done
python3 ~/spd/feed/book.py rm $O; say "AB-EXIT"
