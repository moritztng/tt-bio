#!/bin/bash
# K9 = K8 restarted without b_all (done, ~/r4t/perf/k8/b_all.txt); waits on device holders and named jobs, not on fleet row bookings (card 0 double-booked with vrt-fluid 20:36-02:36Z). K8 on qb2 card 0 (booking fleet-r4t-perf 20:07-02:07Z). (1) bisect which taped-kernel registry entry leaves the
# trimul in-projections without a gradient (K7: TT_BIO_TAPED_KERNELS=all -> dead 61-63, idle device weights 64 vs 50):
# one step on batch00 per entry, plus `all` with the lazy-forward rebuild (R4T_REBUILD_INLINE=0). (2) the PERF protocol
# default arm, 54 steps over the 9 logged batches, coverage guard over one cycle (--cover-steps 9).   k8.sh
C=0; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/k9; mkdir -p $O; cd $WT
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
wait_card() { local h w; for w in $(seq 1 360); do h=$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-"); [ -z "$h" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h" >> $O/run.log; sleep 20; done; echo "ABORT no window $(date -u +%T)" >> $O/run.log; exit 1; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
run() {  # name kernels inline steps batches extra...
  local n=$1 k=$2 il=$3 st=$4 b=$5; shift 5
  wait_card
  taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED TT_BIO_TAPED_KERNELS=$k R4T_REBUILD_INLINE=$il \
    OMP_NUM_THREADS=2 PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1800 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps $st --batch $b \
    --exempt perf/r4t_step/g2_exempt.tsv --out $O/ck "$@" > $O/$n.txt 2>&1
  echo "RC_$n $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
}
B0=$HOME/r4t/step/batch00.npz
for k in reblock_permute_gated pair_transpose tri_att_sdpa_hifi triatt_qkv_heads rne_add rne_add_masked pair_relu; do
  run b_$k $k 1 1 $B0
done
run b_all_lazy all 0 1 $B0
run p_def "" 1 54 $(ls $HOME/r4t/step/batch0?.npz | paste -sd,) --cover-steps 9
kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
