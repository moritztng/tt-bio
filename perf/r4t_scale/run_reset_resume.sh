#!/bin/bash
# Checkpoint/resume through a chip reset, one card, inside this row's book.py window.
# A: 8 uninterrupted steps (ckpt every 2). B: same run SIGKILLed after step 5, the card reset
# with tt-smi -r, then --resume from the step-4 checkpoint to step 8. C (broken arm): resume
# from a copy of B's step-4 checkpoint with the optimizer state restarted. Pass = B's step 5-8
# master digests equal A's, and C's do not. Out ~/r4t/scale/rr/, verdict rr/verdict.json.
set -u
CARD=${1:-2}; OWNER=${2:-fleet-r4t-scale}
WT=$HOME/.coworker/wt/r4t-scale; V=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
cd $WT; export PYTHONPATH=$WT
R=$HOME/r4t/scale/rr; mkdir -p $R
busy() { for f in /proc/[0-9]*/fd/*; do [ "$(readlink $f 2>/dev/null)" = /dev/tenstorrent/$CARD ] && return 0; done; return 1; }
until python3 $HOME/spd/feed/book.py active $CARD 2>/dev/null | grep -q "^$OWNER" && ! busy; do sleep 30; done
export TT_BIO_LEASE_HOLDER=worker:r4t-scale TT_BIO_LEASE_CARDS=$CARD
drv() { timeout 1500 $V perf/r4t_scale/dp_pairstack.py --cards $CARD --steps 8 --every 2 "$@"; }
rm -rf $R/A $R/B $R/C $R/ckA $R/ckB $R/ckC $R/progress $R/verdict.json
drv --ckpt $R/ckA --out $R/A > $R/A.log 2>&1; echo "A rc=$?" >> $R/progress
# B runs in its own session so the kill takes the driver, its rank and the timeout wrapper together:
# killing only the wrapper left the rank holding the card through the reset (11-26 05:01Z).
setsid timeout 1500 $V perf/r4t_scale/dp_pairstack.py --cards $CARD --steps 8 --every 2 --ckpt $R/ckB --out $R/B > $R/B.log 2>&1 & P=$!
until grep -q '"step": 5' $R/B/rank0.log 2>/dev/null || ! kill -0 $P 2>/dev/null; do sleep 0.2; done
kill -9 -- -$P; sleep 2
if busy; then echo "ABORT: card $CARD still open after the kill" >> $R/progress; exit 1; fi
echo "B killed session $P after step 5 at $(date -u +%T), card $CARD closed" >> $R/progress
sleep 2; cp -r $R/ckB $R/ckC; cp $R/B/rank0.json $R/B/rank0.prekill.json
timeout 300 $HOME/.local/bin/tt-smi -r $CARD > $R/reset.log 2>&1; echo "reset rc=$?" >> $R/progress
drv --ckpt $R/ckB --out $R/B --resume > $R/B.resume.log 2>&1; echo "B-resume rc=$?" >> $R/progress
drv --ckpt $R/ckC --out $R/C --resume --resume-weights-only > $R/C.log 2>&1; echo "C rc=$?" >> $R/progress
python3 - $R <<'PY' > $R/verdict.json
import json, sys
R = sys.argv[1]
ld = lambda p: {r["step"]: r for r in json.load(open(p))}
A, B, C, pre = ld(f"{R}/A/rank0.json"), ld(f"{R}/B/rank0.json"), ld(f"{R}/C/rank0.json"), ld(f"{R}/B/rank0.prekill.json")
steps = range(5, 9)
v = {"prekill_steps": sorted(pre), "prekill_match": all(pre[s]["master_digest"] == A[s]["master_digest"] for s in pre),
     "resume_match": all(B[s]["master_digest"] == A[s]["master_digest"] for s in steps),
     "broken_mismatch": all(C[s]["master_digest"] != A[s]["master_digest"] for s in steps),
     "A": {s: A[s]["master_digest"] for s in A}, "B": {s: B[s]["master_digest"] for s in B},
     "C": {s: C[s]["master_digest"] for s in C}, "loss_A8": A[8]["loss"], "loss_B8": B[8]["loss"],
     "aiclk": {k: [min(r["aiclk"][0] for r in X.values()), max(r["aiclk"][1] for r in X.values())] for k, X in (("A", A), ("B", B), ("C", C))}}
v["pass"] = v["prekill_match"] and v["resume_match"] and v["broken_mismatch"]
print(json.dumps(v, indent=1))
PY
echo done >> $R/progress
