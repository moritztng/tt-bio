#!/bin/bash
# K12, qb2 card 0. K10 refuted the nested-node cause. Code read: with reblock_permute_back's entry installed, the
# trimul channel loop's in-place route (out=x_out, c_off) opened under the tape and the entry taped the return value,
# not the caller's buffer, cutting every in-projection gradient. Fix: eligible_back(into=True) refuses a taped operand
# and the entry raises on out=. Arms: reblock_permute_back alone and all on the PRE-fix tree (1a1c53cae, must be
# dead > 7), the same on the fix (must be dead 7), then the PERF protocol arm with all if clean.
C=0; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/k12; mkdir -p $O; cd $WT
PRE=$O/pre; [ -d $PRE ] || git worktree add --detach $PRE 1a1c53cae >/dev/null 2>&1
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
wait_card() { local h w; for w in $(seq 1 360); do h=$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-"); [ -z "$h" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h" >> $O/run.log; sleep 20; done; echo "ABORT no window $(date -u +%T)" >> $O/run.log; exit 1; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
run() {  # name tree kernels steps batches extra...
  local n=$1 tr=$2 k=$3 st=$4 b=$5; shift 5
  wait_card
  taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED TT_BIO_TAPED_KERNELS=$k R4T_REBUILD_INLINE=1 \
    OMP_NUM_THREADS=2 PYTHONPATH=$tr TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1800 $PY $tr/perf/r4t_step/trunk_step.py --n 256 --k 4 --steps $st --batch $b \
    --exempt $tr/perf/r4t_step/g2_exempt.tsv --out $O/ck "$@" > $O/$n.txt 2>&1
  echo "RC_$n $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
}
dead() { grep -o '"dead": [0-9]*' $O/$1.txt | head -1 | grep -o '[0-9]*$'; }
B0=$HOME/r4t/step/batch00.npz
run b_rpb_pre $PRE reblock_permute_back 1 $B0
run b_all_pre $PRE all 1 $B0
run b_rpb_fix $WT reblock_permute_back 1 $B0
run b_all_fix $WT all 1 $B0
echo "DEAD rpb_pre=$(dead b_rpb_pre) all_pre=$(dead b_all_pre) rpb_fix=$(dead b_rpb_fix) all_fix=$(dead b_all_fix)" >> $O/run.log
if [ "$(dead b_all_fix)" = 7 ]; then
  run p_all $WT all 54 $(ls $HOME/r4t/step/batch0?.npz | paste -sd,) --cover-steps 9
else echo "SKIP p_all: all_fix dead $(dead b_all_fix)" >> $O/run.log; fi
kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
