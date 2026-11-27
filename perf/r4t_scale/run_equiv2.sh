#!/bin/bash
# r4t-scale (b), true 2-chip arm: rank 0 on card 0 and rank 1 on card 1 CONCURRENTLY, reduced by
# ProcessAxis and graded against the 1-chip accumulation from the numeric arm (equiv/acc.npz, card 0).
# Card independence: each rank grad must be byte-equal to the same rank run on card 0 (equiv/local_r).
# Run inside a cards-0,1 window (called by run_dp2.sh). Logs in $O.
set -u
WT=$HOME/.coworker/wt/r4t-scale; V=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
O=$HOME/r4t/scale/equiv2; R=$HOME/r4t/scale/equiv; mkdir -p $O; cd $WT; export PYTHONPATH=$WT
echo "start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $O/run.log
cp $R/acc.npz $R/acc.json $O/
for r in 0 1; do
  TT_VISIBLE_DEVICES=$r TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:r4t-scale timeout 1800 \
    $V perf/r4t_scale/dp_equiv.py --role rank --rank $r --n 128 --out $O > $O/rank$r.log 2>&1 &
done
wait; cat $O/rank0.log $O/rank1.log >> $O/run.log
$V perf/r4t_scale/dp_equiv.py --role reduce --k 2 --out $O >> $O/run.log 2>&1
$V perf/r4t_scale/dp_equiv.py --role grade --out $O >> $O/run.log 2>&1; echo "grade rc=$?" >> $O/run.log
cp $O/verdict.json $O/verdict_ok.json
$V - $O $R >> $O/run.log 2>&1 <<"PY"
import sys, json, numpy as np
o, r = sys.argv[1], sys.argv[2]
res = {}
for k in (0, 1):
    a, b = np.load(f"{o}/local_{k}.npz"), np.load(f"{r}/local_{k}.npz")
    res[f"rank{k}_card{k}_vs_card0_bytes_equal"] = sorted(a.files) == sorted(b.files) and all(a[n].tobytes() == b[n].tobytes() for n in a.files)
print("card-independence", json.dumps(res)); open(f"{o}/card_independence.json", "w").write(json.dumps(res))
sys.exit(0 if all(res.values()) else 1)
PY
echo "card-independence rc=$?" >> $O/run.log
$V perf/r4t_scale/dp_equiv.py --role reduce --k 2 --broken --out $O >> $O/run.log 2>&1
$V perf/r4t_scale/dp_equiv.py --role grade --out $O >> $O/run.log 2>&1; echo "broken grade rc=$? (must be 1)" >> $O/run.log
cp $O/verdict.json $O/verdict_broken.json
echo "end $(date -u +%FT%TZ)" >> $O/run.log
bash $HOME/.coworker/wt/r4t-scale/perf/r4t_scale/run_trunk2.sh   # real-step DP points (appended 07:59Z; runs before run_dp2.sh releases the booking)
