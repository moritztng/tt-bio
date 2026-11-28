#!/bin/bash
# D84: step-time cost of attn_fp32 (training default since b9033c291) on the K13 PERF protocol, qb2 card 2.
# Tree = wk/r4t-perf merged with wk/r4t-tape. Arms on/off/off/on interleaved, 30 steps each over the 9 logged
# batches (steps 6..30 counted, 50 per arm), 2 pinned CPUs, AICLK sampled during. Starts after P2 ends, only
# when card 2 is free and the next other booking is >= 35 min away; no start after 23:00Z.
C=2; O=$HOME/r4t/perf/d84; TR=$O/tree; cd $TR
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
GIVE=$(date -u -d "2026-11-27 23:00" +%s)
free() { local h nx; h=$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-r4t-perf$" | grep .)
  [ -n "$h" ] && return 1; [ -n "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return 1
  nx=$(python3 ~/spd/feed/book.py next $C --except fleet-r4t-perf 2>/dev/null | grep -o "^[0-9]*"); [ -n "$nx" ] && [ $((nx - $(date +%s))) -lt ${1:-600} ] && return 1; return 0; }
while ! grep -qE "^(END|GAVE-UP)" $HOME/r4t/perf/p2/run.log; do [ $(date +%s) -gt $GIVE ] && { echo "GAVE-UP p2 $(date -u +%T)" >> $O/run.log; exit 1; }; sleep 30; done
n=0; until free 2100; do [ $(date +%s) -gt $GIVE ] && { echo "GAVE-UP $(date -u +%T)" >> $O/run.log; exit 1; }; n=$((n+1)); [ $n = 1 ] && echo "WAIT $(date -u +%T)" >> $O/run.log; sleep 30; done
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
for r in on_1 off_1 off_2 on_2; do
  until free 300; do sleep 20; done
  taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED -u TT_BIO_TAPED_KERNELS R4T_REBUILD_INLINE=1 \
    OMP_NUM_THREADS=2 PYTHONPATH=$TR TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 $PY $TR/perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 30 --batch $B \
    --exempt $TR/perf/r4t_step/g2_exempt.tsv --out $O/ck --cover-steps 9 --attn-fp32 ${r%_*} > $O/$r.txt 2>&1
  echo "RC_$r $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
done
kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
