#!/bin/bash
# Precision arm of the Mol-1 DockQ A/B (mol1-dockq-prec) on qb1 or qb2: the cases of a CASES.json folded with tree SHA
# under the lever set LEVERS (TT_BIO_LEVERS, which also drops size_levers), seed-major so every finished seed is a
# paired read against the default arm. Each card folds only inside its own book.py window and rebooks when one ends.
#   bash prec_arm.sh SHA OUT CASES.json LEVERS CARD [CARD ...]     log on stdout, PREC-EXIT at the end
set -uo pipefail
SHA=${1:?sha}; OUT=${2:?out}; CASES=$(realpath "${3:?cases}"); LEV=${4:?levers}; shift 4; [ $# -gt 0 ] || { echo "no card"; exit 2; }
cd ~/mol1-quality; say() { echo "$(date -u +%FT%TZ) $*"; }
case $(hostname) in tt-quietbox2) LOCK_HOST=qb2 ;; tt-quietbox) LOCK_HOST=qb1 ;; *) say "unknown host"; exit 2 ;; esac
[ -f main-$SHA/.ok ] || { say "no tree main-$SHA (bh_arm.sh/ab_arm.sh unpack it)"; exit 1; }
export PATH=$HOME/tt-bio-dev/env/bin:$HOME/mol1-quality/env/kalign/bin:$PATH PYTHONPATH=$PWD/main-$SHA LOCK_HOST
export TT_BIO_LEVERS=$LEV TT_BIO_LEASE_HOLDER=worker:mol1-dockq-prec
mkdir -p "$OUT"; echo "$SHA" > "$OUT/ENGINE_SHA"
python3 -c 'import tt_bio.tenstorrent as T; print(",".join(sorted(T.model_levers())))' > "$OUT/LEVERS" 2>/dev/null
say "engine $SHA, levers $(cat "$OUT/LEVERS"), cases $CASES, cards $*"
for c in "$@"; do
  ( o=mol1-dockq-prec-c$c
    for s in 42 66 101 2024 8888; do
      while :; do
        until [ "$(python3 ~/spd/feed/book.py active "$c" 2>/dev/null)" = "$o" ]; do
          python3 ~/spd/feed/book.py add $o "$c" 14400 --note "mol1-dockq-prec levers=$LEV ($SHA)" > /dev/null; sleep 60; done
        say "card $c: window open, seed $s"
        bash ~/mol1-quality/code/run_tt.sh --book $o --cases "$CASES" inputs "$OUT" "$s" "$c" > "$OUT.c$c.last" 2>&1
        cat "$OUT.c$c.last"
        grep -q "TT-STOP\|card lock held" "$OUT.c$c.last" || break
        sleep 60
      done
    done
    python3 ~/spd/feed/book.py rm $o > /dev/null; say "card $c: done" ) &
done
wait; say "PREC-EXIT"
