#!/bin/bash
# D2 gate, qb2 card 0 after K13 ends. base = f9083dfd9 (pre-D2) vs d2 = this tree; 9 steps over the 9 batches each,
# --pcie-meter on. Pass: master digest equal at every step, winners equal, fwd up_MB down by ~302 MB.
C=0; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/d2; mkdir -p $O; cd $WT
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
until grep -q "^END" $HOME/r4t/perf/k13/run.log; do sleep 30; done
[ -d $O/base ] || git worktree add --detach $O/base f9083dfd9 >> $O/run.log 2>&1
wait_card() { local h w; for w in $(seq 1 360); do h=$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-"); [ -z "$h" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h" >> $O/run.log; sleep 20; done; echo "ABORT no window $(date -u +%T)" >> $O/run.log; exit 1; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
for arm in base d2; do
  tr=$WT; [ $arm = base ] && tr=$O/base
  wait_card
  taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED -u TT_BIO_TAPED_KERNELS R4T_REBUILD_INLINE=1 \
    OMP_NUM_THREADS=2 PYTHONPATH=$tr TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1800 $PY $WT/perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 9 --batch $B \
    --exempt $WT/perf/r4t_step/g2_exempt.tsv --out $O/ck --pcie-meter on > $O/$arm.txt 2>&1
  echo "RC_$arm $? $(date -u +%T)" >> $O/run.log; rm -rf $O/ck
done
kill $CL
python3 - $O >> $O/run.log <<"PY"
import json, sys
o = sys.argv[1]
r = {a: [json.loads(l) for l in open(f"{o}/{a}.txt") if l.startswith("{\"step\"")] for a in ("base", "d2")}
same = [x["master_digest"] == y["master_digest"] and x["winner"] == y["winner"] for x, y in zip(r["base"], r["d2"])]
print("DIGESTS", "PASS" if same and all(same) and len(same) == 9 else "FAIL", same)
for a in r:
    print(a, "fwd_up_MB", [x["pcie"]["fwd"]["up_MB"] for x in r[a]], "fwd_s", [x["fwd_s"] for x in r[a]])
PY
echo "END $(date -u +%FT%TZ)" >> $O/run.log
