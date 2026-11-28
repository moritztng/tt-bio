# K6 screen: TT_BIO_TAPED_KERNELS=all (tape entries for the fused kernels) vs default, trunk_step + ksplit. Not a grade.
cd ~/.coworker/wt/r4t-perf; O=~/r4t/perf/k6; mkdir -p $O
E="env -u TT_BIO_LEVERS OMP_NUM_THREADS=2 PYTHONPATH=$PWD TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1"
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
A="perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $HOME/r4t/step/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv"
echo "START $(date -u +%T) sha $(git rev-parse --short HEAD)" > $O/run.log
taskset -c 6,7 $E TT_BIO_TAPED_KERNELS=all KSPLIT_REPS=3 KSPLIT_OUT=$O/ks_all.json timeout 600 $PY perf/r4t_perf/ksplit.py $A --steps 2 --out $O/ck > $O/ks_all.log 2>&1; echo "RC_ks $? $(date -u +%T)" >> $O/run.log
for i in 1 2; do for arm in off all; do
  if [ $arm = all ]; then X="TT_BIO_TAPED_KERNELS=all"; else X="TT_BIO_TAPED_KERNELS="; fi
  taskset -c 6,7 $E $X timeout 900 $PY $A --steps 3 --out $O/ck > $O/t_${arm}$i.txt 2>&1; echo "RC_t_${arm}$i $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
done; done
echo END $(date -u +%T) >> $O/run.log
