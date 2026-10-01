#!/bin/bash
# Gradient sweep for the query-chunked triatt_bw, qb2 card 0.
cd /home/ttuser/.coworker/wt/bcw-dbias
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcw-dbias
PY=~/tt-bio-dev/env/bin/python
O=perf/bcw_dbias/out
run() { tag=$1; shift; echo "== $tag $(date -u +%T)"; timeout 900 $PY perf/bcw_dbias/grade.py "$@" --out $O/$tag.json > $O/$tag.log 2>&1; echo "rc=$? $(date -u +%T)"; }
run n128_whole --b 8 --n 128
run n288_whole --b 16 --n 288
run n288_qt3 --b 16 --n 288 --qt 3
run n512_serve --b 16 --n 512
run n544_serve --b 16 --n 544
run n768_serve --b 16 --n 768
run n864_serve --b 8 --n 864
run n288_b288_serve --b 288 --n 288 --no-fallback
echo ALLDONE
