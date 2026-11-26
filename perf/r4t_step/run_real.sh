#!/bin/bash
# r4t-step: the real RFD4 step on upstream's dumped crop-256 batches, on CARD $1 inside booking $OWNER.
# B: 2-step smoke on batch00. If it fails, the synthetic-input arms (run_trunk.sh A/R/D) run instead.
# A: 6 steps, checkpoint at 5. R: fresh process resumes from A's step 5, step 6 must match. D: layer 3 detached, G2 must refuse.
cd ~/.coworker/wt/r4t-step || exit 1
C=${1:-1}; OUT=~/r4t/step/real; mkdir -p $OUT; S=$HOME/r4t/step; B=${BATCH:-$S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz,$S/batch03.npz,$S/batch06.npz,$S/batch02.npz,$S/batch04.npz,$S/batch07.npz}
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
ok=
for i in $(seq 1 300); do
  who=$(python3 ~/spd/feed/book.py active $C 2>/dev/null)
  if [ "$who" = "${OWNER:-fleet-r4t-step}" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ]; then ok=1; break; fi
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> $OUT/wait.log; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=${GRANT:-$C} TT_BIO_LEASE_HOLDER=worker:r4t-step timeout 2400 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B "$@"; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
rm -rf $OUT/ckB $OUT/ckA $OUT/ckR $OUT/ckD
R --steps 2 --out $OUT/ckB > $OUT/B.log 2>&1; rc=$?; echo "RC_B $rc $(date -u +%T)" >> $OUT/run.log
if [ $rc != 0 ]; then
  echo "real smoke failed; synthetic arms instead" >> $OUT/run.log
  exec bash perf/r4t_step/run_trunk.sh $C
fi
R --steps 6 --save-at 5 --out $OUT/ckA > $OUT/A.log 2>&1; echo "RC_A $? $(date -u +%T)" >> $OUT/run.log
R --steps 6 --resume $OUT/ckA/adapter-00000005.safetensors --out $OUT/ckR > $OUT/R.log 2>&1; echo "RC_R $? $(date -u +%T)" >> $OUT/run.log
R --steps 6 --detach-at 3 --out $OUT/ckD > $OUT/D.log 2>&1; echo "RC_D $? $(date -u +%T)" >> $OUT/run.log
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
