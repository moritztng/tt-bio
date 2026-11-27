#!/bin/bash
# P2: winner pair on card. Block grade (normal, broken, 2nd batch) then trunk_step A/B host vs card, interleaved, qb2 card 2.
C=2; OW=r4t-perf-p2; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/p2; cd $WT
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
# Runs only in card 2's free gap inside the fleet-r4t-perf booking: after the booking ahead of it (ends 11:32Z), done before
# dfd-perf-wlt (13:32Z). ~45 min of work, so no start after 12:40Z.
NB=$(date -u -d "2026-11-27 11:32" +%s); NA=$(date -u -d "2026-11-27 12:40" +%s)
for w in $(seq 1 1080); do [ $(date +%s) -gt $NA ] && { echo "GAVE-UP $(date -u +%T) past 12:40Z" >> $O/run.log; exit 1; }; [ $(date +%s) -lt $NB ] && { sleep 20; continue; }; h=$(python3 ~/spd/feed/book.py active $C --except $OW 2>/dev/null | sed 's/ *fleet-r4t-perf//'); [ -z "$(echo $h | tr -d ' ')" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; [ $w = 1080 ] && { echo GAVE-UP >> $O/run.log; exit 1; }; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h" >> $O/run.log; sleep 20; done
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
run() { taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED -u TT_BIO_TAPED_KERNELS R4T_REBUILD_INLINE=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$WT WT=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
  TT_BIO_SDPA_OWN_FORWARD=1 "$@"; }
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
run timeout 900 $PY perf/r4t_perf/pair_vjp_card.py $HOME/r4t/step/batch00.npz 0 > $O/g00.txt 2>&1; echo "G00 rc $? $(date -u +%T)" >> $O/run.log
BREAK=1 run timeout 900 $PY perf/r4t_perf/pair_vjp_card.py $HOME/r4t/step/batch00.npz 0 > $O/gbrk.txt 2>&1; echo "GBRK rc $? $(date -u +%T)" >> $O/run.log
run timeout 900 $PY perf/r4t_perf/pair_vjp_card.py $HOME/r4t/step/batch03.npz 1 > $O/g03.txt 2>&1; echo "G03 rc $? $(date -u +%T)" >> $O/run.log
for rep in 1 2; do for arm in host card; do
  run timeout 1500 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 9 --cover-steps 9 --batch $B \
    --exempt perf/r4t_step/g2_exempt.tsv --out $O/ck --pcie-meter on --win-pair $arm > $O/ab_${arm}_$rep.txt 2>&1
  echo "AB $arm $rep rc $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
done; done
echo "END $(date -u +%T)" >> $O/run.log; kill $CL; python3 ~/spd/feed/book.py rm $OW
