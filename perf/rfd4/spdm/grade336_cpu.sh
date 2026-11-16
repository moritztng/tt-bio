#!/bin/bash
# grade336_cpu.sh OUT SEEDS OURS [REF]  : card-free grade leg for grade336.sh's designs. Pairs OURS against REF
# seed for seed (perf/rfd4/e2e/grade.py: designed-chain CA RMSD after superposing on the conditioned target),
# takes the seed floor from REF at the next seed, then runs the oracle-free binder grade over the two pools.
# Summary lands in OUT/SUMMARY.txt, the binder verdict in OUT/quality-OURS.json.
set -u
OUT=$1 N=$2 OURS=$3 REF=${4:-base}
WT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$WT"; export PYTHONPATH=$WT:$WT/perf
OUT=$(cd "$OUT" && pwd)
s() { ls "$OUT/$1"/*_s$2_r*_generated.cif 2>/dev/null | tail -1; }
mkdir -p "$OUT/grade-$OURS" "$OUT/pool-$OURS/tt" "$OUT/pool-$OURS/ref"
for k in $(seq 0 "$N"); do
  o=$(s "$OURS" "$k"); r=$(s "$REF" "$k"); f=$(s "$REF" $(( (k+1) % (N+1) )))
  [ -z "$o" ] || [ -z "$r" ] || [ -z "$f" ] && { echo "seed $k missing a cif" >> "$OUT/done"; continue; }
  ln -sf "$o" "$OUT/pool-$OURS/tt/"; ln -sf "$r" "$OUT/pool-$OURS/ref/"
  nice -n 19 python perf/rfd4/e2e/grade.py --ours "$o" --upstream "$r" --floor "$f" \
    --out "$OUT/grade-$OURS/s$k.json" > /dev/null 2>&1 || echo "grade $OURS s$k failed" >> "$OUT/done"
done
nice -n 19 python perf/rfd4/quality.py grade --feature binder --tt "$OUT/pool-$OURS/tt" --ref "$OUT/pool-$OURS/ref" \
  --allow-no-oracle --label "spd-m-rfd4 $OURS vs $REF, 336 tokens, $((N+1)) seeds" \
  --out "$OUT/quality-$OURS.json" > "$OUT/quality-$OURS.log" 2>&1
echo "quality rc=$?" >> "$OUT/done"
python - "$OUT/grade-$OURS" <<'PY' >> "$OUT/SUMMARY.txt"
import json, sys, glob, statistics
rs = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/s*.json"))]
if rs:
    d = [r["rmsd_a"] for r in rs]; fl = [r["seed_floor_a"] for r in rs]
    print(f"{sys.argv[1]}: n={len(d)} designed-chain CA RMSD median {statistics.median(d):.2f} A "
          f"(max {max(d):.2f}) against a seed floor median {statistics.median(fl):.2f} A, "
          f"ratio {statistics.median(d)/statistics.median(fl):.3f} (kill line 0.5)")
PY
