#!/bin/bash
# After run.sh: foldprof default vs --rm (A, RM, A2, RM2), 3 steps on batch00..02, card 0.
while pgrep -f "^/bin/bash ./run.sh" >/dev/null; do sleep 30; done
S=$HOME/r4t/perf/bis/src; O=$HOME/r4t/perf/fold; mkdir -p $O
cp $HOME/.coworker/wt/r4t-perf/perf/r4t_perf/foldprof.py $S/perf/r4t_perf/
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python; D=$HOME/r4t/step; cd $S
for arm in A RM A2 RM2; do
  f=; case $arm in RM*) f=--rm;; esac
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/0 2>/dev/null)" ] && break; sleep 20; done
  env PYTHONPATH=$S TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 \
    timeout 1500 $PY perf/r4t_perf/foldprof.py $f -- --n 256 --k 4 --steps 3 --batch $D/batch00.npz,$D/batch01.npz,$D/batch02.npz \
    --exempt perf/r4t_step/g2_exempt.tsv --out $O/ck_$arm > $O/$arm.log 2>&1
  echo "RC_$arm $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck_$arm
done
