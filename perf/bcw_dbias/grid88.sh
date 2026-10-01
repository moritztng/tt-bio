#!/bin/bash
# After t768.sh on card 1: the looped kernel on Wormhole 8x8 work split with a card open, at the
# chunk Wormhole would take, graded against float64 and against the fallback.
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; log=$out/grid88.log
until grep -q "T768 DONE" $out/t768.log 2>/dev/null; do sleep 30; done
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcw-dbias
G() { tag=$1; shift; echo "=== $(date -u +%FT%TZ) $tag" >> $log
      timeout 1200 ~/tt-bio-dev/env/bin/python perf/bcw_dbias/grade.py --grid 8x8 "$@" --out $out/$tag.json > $out/$tag.log 2>&1
      echo "=== rc=$? $tag" >> $log; }
G g88_n480_qt3 --b 64 --n 480 --qt 3
G g88_n544 --b 64 --n 544
G g88_n768 --b 48 --n 768
G g88_n800 --b 32 --n 800
echo "=== GRID88 DONE $(date -u +%FT%TZ)" >> $log
