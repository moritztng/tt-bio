#!/bin/bash
# D2 gate, second run. d2.sh omitted --cover-steps 9, so the 5-step dead-param guard stopped both arms identically at
# step 5 (its first 5 digests already agreed). Arms on qb2 card 0: base (f9083dfd9, pre-D2) vs d2 (this tree), 9 steps
# over the 9 batches; discrimination arm `lnoff` (d2 tree, --lnbw-weights off) must DIFFER from base, so the compare can
# FAIL; serving A/B: `tt-bio design --model rfd4` on the served 336-token fixture, 2 designs seed 7, both trees, ATOM-line
# sha256 per CIF must match.
C=${C:-0}; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/d2b; mkdir -p $O; cd $WT
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python; BASE=$HOME/r4t/perf/d2/base
wait_card() { local h w; for w in $(seq 1 360); do h=$(python3 ~/spd/feed/book.py active $C --except fleet-r4t-perf 2>/dev/null | tr " " "\n" | grep -v "^fleet-"); [ -z "$h" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h" >> $O/run.log; sleep 20; done; echo "ABORT no window $(date -u +%T)" >> $O/run.log; exit 1; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
step() {  # name tree extra...
  local n=$1 tr=$2; shift 2
  wait_card
  taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED -u TT_BIO_TAPED_KERNELS R4T_REBUILD_INLINE=1 \
    OMP_NUM_THREADS=2 PYTHONPATH=$tr TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1800 $PY $WT/perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 9 --batch $B \
    --exempt $WT/perf/r4t_step/g2_exempt.tsv --cover-steps 9 --out $O/ck --pcie-meter on "$@" > $O/$n.txt 2>&1
  echo "RC_$n $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
}
serve() {  # name tree
  local n=$1 tr=$2; wait_card; rm -rf $O/srv_$n
  env -u TT_BIO_LEVERS PYTHONPATH=$tr TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    timeout 2400 $PY -m tt_bio.main design $WT/perf/dsfix/fixtures/rfd4_S336.yaml --model rfd4 --num_designs 2 --seed 7 \
    --out_dir $O/srv_$n > $O/srv_$n.txt 2>&1
  echo "RC_srv_$n $? $(date -u +%T)" >> $O/run.log
  find $O/srv_$n -name "*.cif" | sort | while read f; do echo "$(basename $f) $(grep -E '^(ATOM|HETATM)' $f | sha256sum | cut -c1-16)"; done > $O/srv_$n.sha
}
# `d2b.sh base` reruns only the base arm (the first run lost card 0 to an spd lease at device open), then re-grades.
# `C=2 d2b.sh srv` reruns only the serving pair on card C (the first run died after srv_base: the script was edited
# while bash was still reading it). srv_base from card 0 is kept as srv_base_c0.sha, a cross-card check.
if [ "${1:-}" = base ]; then step base $BASE
elif [ "${1:-}" = srv ]; then [ -f $O/srv_base.sha ] && mv $O/srv_base.sha $O/srv_base_c0.sha; serve base $BASE; serve d2 $WT; else
step base $BASE
step d2 $WT
step lnoff $WT --lnbw-weights off
serve base $BASE
serve d2 $WT
fi
kill $CL
python3 - $O >> $O/run.log <<"PY"
import json, sys
o = sys.argv[1]
r = {a: [json.loads(l) for l in open(f"{o}/{a}.txt") if l.startswith("{\"step\"")] for a in ("base", "d2", "lnoff")}
def cmp(a, b):
    s = [x["master_digest"] == y["master_digest"] and x["winner"] == y["winner"] for x, y in zip(r[a], r[b])]
    return s, len(s) == 9 and all(s)
s, ok = cmp("base", "d2"); print("DIGESTS", "PASS" if ok else "FAIL", s)
s, ok = cmp("base", "lnoff"); print("DISCRIM", "PASS (differs)" if not all(s) and len(s) == 9 else "FAIL (cannot fail)", s)
for a in ("base", "d2"):
    print(a, "fwd_up_MB", [x["pcie"]["fwd"]["up_MB"] for x in r[a]], "fwd_s", [x["fwd_s"] for x in r[a]], "step_s", [x.get("step_s") for x in r[a]])
sb, sd = (open(f"{o}/srv_{a}.sha").read() for a in ("base", "d2"))
print("SERVING", "PASS" if sb and sb == sd else "FAIL", sb.split(), sd.split())
import os
if os.path.exists(f"{o}/srv_base_c0.sha"):
    c0 = open(f"{o}/srv_base_c0.sha").read(); print("SERVING-CROSS-CARD", "PASS" if c0 == sb else "FAIL", c0.split())
PY
echo "END $(date -u +%FT%TZ)" >> $O/run.log
