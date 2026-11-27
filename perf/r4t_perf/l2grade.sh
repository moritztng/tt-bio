#!/bin/bash
# Grade l2t.sh output: D43 parity of the ON arm (grade_parity --floor), drift ON vs OFF (r2cmp), off-arm parity for
# reference, then timing medians of steps 2-3 per arm.   l2grade.sh [OUT]
O=${1:-$HOME/r4t/perf/l2t}; WT=$HOME/.coworker/wt/r4t-perf; cd $WT
for arm in ON OFF; do
  python3 perf/r4t/parity_from_step.py $O/${arm}_batch0?.log --dtype bfloat16 --out $O/parity_$arm.json > /dev/null
  echo "== $arm"; $HOME/scratch/shipflow/venv-c4f99e695d34/bin/python perf/r4t/grade_parity.py --parity $O/parity_$arm.json --floor perf/r4t/floor/up_bf16.json | head -8 | cut -c1-400
done
echo "== drift ON vs OFF"; python3 perf/r4t_perf/r2/r2cmp.py $O ON $O OFF
python3 - "$O" <<'PY'
import json, statistics as st, sys
d = sys.argv[1]
for arm in "AB":
    R = []
    for i in (1, 2, 3):
        L = [json.loads(x) for x in open(f"{d}/{arm}{i}.log") if x.startswith('{"step"')]
        R += [r for r in L if r["step"] >= 2]
        print(arm, i, [r["master_digest"][:8] for r in L], "aiclk", min(r["aiclk"][0] for r in L), max(r["aiclk"][1] for r in L), L[-1].get("lnbw_reach"))
    print(arm, "medians", {k: round(st.median(r[k] for r in R), 3) for k in ("step_s", "fwd_s", "bwd_s", "fold_bwd_s", "opt_s", "rebuild_s")}, "n", len(R))
PY
