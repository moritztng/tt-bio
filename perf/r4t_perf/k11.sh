#!/bin/bash
# K11, qb2 card 0: leave-one-out over the taped-kernel registry. K10 refuted the nested-node cause (reblock_permute alone is clean pre-fix, all still dead 63), and every entry alone is clean, so the dead in-projection grads need two entries together. Arm all-minus-X clean => X is in the pair. kernels= in the JSON shows which entries served.
# `taped_ttnn._permute_fwd` calling the fused `reblock_permute` wrapper while recording: with the registry's
# `reblock_permute` entry installed it records a second node over an untracked leaf. Arms: (1) `reblock_permute`
# alone on the PRE-fix tree (must FAIL: dead > 7), (2) same entry and (3) `all` on the fixed tree (must equal
# default: dead 7, idle_dev_w 50), (4) if (3) passes, the PERF protocol arm with `all`, 54 steps / 9 batches.
C=0; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/k11; mkdir -p $O; cd $WT
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
ALL="tri_att_sdpa_hifi reblock_permute reblock_permute_back reblock_permute_gated pair_transpose triatt_qkv_heads rne_add rne_add_masked pair_relu"
run b_rpb $WT reblock_permute_back 1 $B0
run b_all $WT all 1 $B0
for x in $ALL; do run lo_$x $WT $(echo $ALL | tr " " "\n" | grep -vx $x | paste -sd,) 1 $B0; done
for f in b_rpb b_all $(for x in $ALL; do echo lo_$x; done); do echo "DEAD $f $(dead $f)" >> $O/run.log; done
kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
