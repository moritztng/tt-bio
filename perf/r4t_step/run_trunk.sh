#!/bin/bash
# r4t-step: inside our qb2 booking on CARD ($1, default 2) (book.py owner fleet-r4t-step), a crop-128 smoke first, then the
# trunk-step arms A (6 steps, save at 5), R (resume, step 6 must match A), D (detached layer: G2 must refuse).
# Never opens the card outside the booking: a wait that times out exits.
cd ~/.coworker/wt/r4t-step || exit 1
C=${1:-2}; OUT=~/r4t/step/trunk; mkdir -p $OUT
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
ok=
# my window: $OWNER if given, else any r4t-step booking (dispatcher fleet-r4t-step or a hand-made r4t-step-*)
mine(){ if [ -n "$OWNER" ]; then [ "$1" = "$OWNER" ]; else case "$1" in fleet-r4t-step|r4t-step-*) return 0;; esac; return 1; fi; }
for i in $(seq 1 300); do
  who=$(python3 ~/spd/feed/book.py active $C 2>/dev/null)
  if mine "$who" && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ]; then ok=1; break; fi
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> $OUT/wait.log; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${GRANT:-$C} TT_BIO_LEASE_HOLDER=worker:r4t-step timeout 1500 $PY perf/r4t_step/trunk_step.py "$@"; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
rm -rf $OUT/ckS $OUT/ckA $OUT/ckR $OUT/ckD
R --n 128 --k 2 --depth 2 --steps 2 --out $OUT/ckS > $OUT/S.log 2>&1; rc=$?; echo "RC_S $rc" >> $OUT/run.log
[ $rc = 0 ] || { echo "END smoke failed $(date -u +%FT%TZ)" >> $OUT/run.log; exit 3; }
R --n 256 --k 4 --depth 2 --steps 6 --save-at 5 --out $OUT/ckA > $OUT/A.log 2>&1; echo "RC_A $?" >> $OUT/run.log
R --n 256 --k 4 --depth 2 --steps 6 --resume $OUT/ckA/adapter-00000005.safetensors --out $OUT/ckR > $OUT/R.log 2>&1; echo "RC_R $?" >> $OUT/run.log
R --n 256 --k 1 --depth 2 --steps 6 --detach-at 3 --out $OUT/ckD > $OUT/D.log 2>&1; echo "RC_D $?" >> $OUT/run.log
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
