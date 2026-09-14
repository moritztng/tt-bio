#!/bin/bash
# pass 3 chain: finish the 44-leg gate, score it, then take the benchlocked fold A/B.
#
# The box cannot go quiet while this row own 44-leg gate is folding, so the A/B has to be
# chained behind it rather than raced against it. Every step writes into this worktree.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
cd $W
export PYTHONPATH=$W
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
PY=/home/ttuser/tt-bio-dev/env/bin/python3
GATEPID=${1:-0}
L=$W/perf/b2z2_zinit_ship/pass3_chain.log
say(){ echo "[$(date -Is)] $*" >> $L; }

say "chain start, waiting on gate pid $GATEPID"
while [ "$GATEPID" != "0" ] && kill -0 "$GATEPID" 2>/dev/null; do sleep 60; done
say "gate pid gone"

# Resume the gate until it has all 44 legs. Resumable per leg, so a re-run only pays for what is
# missing. Three attempts: contention timeouts (protenix-hsa-msa, protenix-v1-prot-msa) are what
# this is for, and a leg that fails three times is a real failure, not a flake.
for try in 1 2 3; do
  n=$($PY - <<PYEOF
import json,pathlib
p=pathlib.Path("perf/b2z2_gate/gate_zinitship.json")
try: print(len(json.loads(p.read_text())["legs"]))
except Exception: print(0)
PYEOF
)
  say "gate report has $n/44 legs (attempt $try)"
  [ "$n" -ge 44 ] && break
  say "resuming gate"
  TT_BIO_LEASE_CARDS=3 $PY -u scripts/full_parity_gate.py --workers localhost:3 \
    --workdir $W/perf/b2z2_gate/zinitship_work \
    --out $W/perf/b2z2_gate/gate_zinitship.json >> $W/perf/b2z2_gate/gate_zinitship.log 2>&1
  say "gate resume attempt $try exited $?"
done

say "scoring gate"
$PY perf/b2z2_qchunk_ship/gate_compare.py --ship perf/b2z2_gate/gate_zinitship.json \
  > $W/perf/b2z2_zinit_ship/gate_compare_ship.txt 2>&1
say "gate_compare exited $? -> perf/b2z2_zinit_ship/gate_compare_ship.txt"

say "queuing the benchlocked fold A/B on card 3"
BENCHLOCK_WAIT_S=10800 BENCHLOCK_LOAD_WAIT_S=3600 \
  /home/ttuser/.coworker/scripts/benchlock.sh b2z2-zinit-ship -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship \
  $PY -u perf/b2z2_cond/fold_cond.py --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --sizes 512 --timing-reps 8 --timing-arms base,on,base \
  --out $W/perf/b2z2_zinit_ship/fold_ab512_qb2_c3.json \
  --cifdir $W/perf/b2z2_zinit_ship/ab_cifs >> $W/perf/b2z2_zinit_ship/fold_ab512.log 2>&1
say "fold A/B exited $? (75 = benchlock timed out, do NOT read a number out of it)"
say "CHAINDONE"
