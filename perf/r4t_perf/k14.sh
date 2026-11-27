#!/bin/bash
# K14 (r4t-perf): grad-off K-candidate pass runs outside ag.tape() under ag.suspended() with trunk._levers
# swapped to include pair_bf16z/dit_sigf32/tok_recycle_mm (--cand-levers serve), unlocking the shipped
# serving fused kernels (trimul_t1, pair_tail_cm, pair_relu, outer_residual, rms_pair_bias, rne_add).
# Pre-registered bar (2026-11-27 17:16Z, before any card run):
#   1. diag_serve: TRIMUL_STATS t1[0] > 0 (T1 actually served)
#   2. Paired ABBA (serving_1, serve_1, serve_2, serving_2, 18 steps = 2 cycles of batch00..08):
#      winners 18/18 identical, master_digest 18/18 identical, dead & idle_dev_w identical,
#      AICLK >= 1200 MHz, clear fwd_s win.
C=0; O=$HOME/r4t/perf/k14; TR=$HOME/.coworker/wt/r4t-perf; mkdir -p $O; cd $TR
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
GIVE=$(date -u -d "2026-11-27 23:00" +%s)
free() {
  grep -qE "^(END|GAVE-UP)" $HOME/r4t/perf/c1/run.log 2>/dev/null || return 1
  [ -n "$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-r4t-perf$" | grep .)" ] && return 1
  [ -n "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return 1
  [ -n "$(python3 ~/spd/feed/book.py active $C 2>/dev/null | grep -o fleet-r4t-perf)" ] || return 1
  return 0
}
n=0; until free; do
  [ $(date +%s) -gt $GIVE ] && { echo "GAVE-UP $(date -u +%T)" >> $O/run.log; exit 1; }
  n=$((n+1)); [ $n = 1 ] && echo "WAIT $(date -u +%T)" >> $O/run.log
  sleep 30
done
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d" " -f1 /proc/loadavg)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
run() { taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED -u TT_BIO_TAPED_KERNELS R4T_REBUILD_INLINE=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$TR WT=$TR TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
  TT_BIO_SDPA_OWN_FORWARD=1 "$@"; }
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
step() { run timeout 1800 $PY $TR/perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B --exempt $TR/perf/r4t_step/g2_exempt.tsv \
  --out $O/ck --cover-steps 9 "$@"; }

# 1. 2-step diagnostic with TRIMUL_STATS=1
TT_BIO_TRIMUL_STATS=1 step --steps 2 --cand-levers serve > $O/diag_serve.txt 2>&1; echo "RC_diag_serve $? $(date -u +%T)" >> $O/run.log; rm -rf $O/ck
TT_BIO_TRIMUL_STATS=1 step --steps 2 --cand-levers serving > $O/diag_serving.txt 2>&1; echo "RC_diag_serving $? $(date -u +%T)" >> $O/run.log; rm -rf $O/ck
grep -h "^TRIMUL_STATS" $O/diag_serve.txt $O/diag_serving.txt >> $O/run.log || true

# 2. Paired ABBA (18 steps = 2 cycles of batch00..08)
for r in serving_1 serve_1 serve_2 serving_2; do
  step --steps 18 --cand-levers ${r%_*} > $O/$r.txt 2>&1
  echo "RC_$r $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $O/run.log
  rm -rf $O/ck
done

# 3. Combined P2 + K14 preview (18 steps) if N1 P2 passed
if grep -q "GRADE g00=PASS gbrk=FAIL g03=PASS" $HOME/r4t/perf/n1/run.log 2>/dev/null; then
  step --steps 18 --win-pair card --cand-levers serve > $O/combo_p2_k14.txt 2>&1
  echo "RC_combo $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $O/run.log
  rm -rf $O/ck
fi

kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
