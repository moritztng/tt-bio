#!/bin/bash
# rfd4-wh-bringup: grade the Wormhole designs against Blackhole's, seed by seed, on the CPU (pc).
# Same two instruments LAND9-12 used: grade.py per seed (designed-chain CA RMSD, WH vs BH at the same seed, against the
# BH seed floor: BH at seed k vs BH at seed k+1) and quality.py's population grade on the 31-design pools.
#   bash perf/rfd4/wh/grade.sh <size>          pulls .103's seeds<size> and qb2's br<size> into artifacts/wh/
# Also checks the branch against main on Blackhole: every br<size> seed main also ran must be byte-identical (ATOM).
set -u
N=${1:?size}; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
PY=${PY:-/home/moritz/rfd4-cpu-venv/bin/python}; export PYTHONPATH=$WT
A=$WT/artifacts/wh; mkdir -p $A/wh$N $A/bh$N $A/mn$N
rsync -a 172.16.102.103:spd/rfd4-wh/out/seeds$N/designs/ $A/wh$N/
rsync -a tt-quietbox2:scratch/rfd4-wh-bringup/runs/br$N/ $A/bh$N/
rsync -a tt-quietbox2:scratch/rfd4-wh-bringup/runs/mn$N/ $A/mn$N/ 2>/dev/null
f() { ls $A/$1/pdl1_binder_${N}_s$2_r$(($2+1))_generated.cif 2>/dev/null; }
G=$A/grade$N; P=$A/pool$N; rm -rf $G $P; mkdir -p $G $P/tt $P/ref
for k in $(seq 0 30); do
  w=$(f wh$N $k); b=$(f bh$N $k); fl=$(f bh$N $(( (k+1)%31 )))
  [ -n "$w" ] && [ -n "$b" ] && [ -n "$fl" ] || { echo "s$k missing (wh=${w:+y} bh=${b:+y} floor=${fl:+y})"; continue; }
  ln -sf $w $P/tt/; ln -sf $b $P/ref/
  $PY perf/rfd4/e2e/grade.py --ours $w --upstream $b --floor $fl --out $G/s$k.json > /dev/null 2>&1 || echo "grade s$k failed"
done
$PY - $G <<'PY'
import json, sys, glob, statistics as st
rows = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1] + "/s*.json"))]
r = [x["rmsd_a"] for x in rows]; fl = [x["seed_floor_a"] for x in rows]
print(f"n {len(rows)} | WH vs BH CA RMSD median {st.median(r):.2f} A (p25 {sorted(r)[len(r)//4]:.2f}, max {max(r):.2f}) "
      f"| BH seed floor median {st.median(fl):.2f} A (min {min(fl):.2f}) | under floor {sum(a < b for a, b in zip(r, fl))}/{len(r)}")
PY
# quality.py reads the seed floor from its --out doc; seed it with the calibrated binder floor LAND9-12 graded against
$PY -c 'import json, sys; json.dump({"seed_floor": json.load(open(sys.argv[1]))["seed_floor"]}, open(sys.argv[2], "w"))' \
  /home/moritz/.coworker/state/rfd4/QUALITY.json $A/quality$N.json
$PY perf/rfd4/quality.py grade --feature binder --tt $P/tt --ref $P/ref --allow-no-oracle \
  --label "rfd4-wh-bringup: Wormhole vs Blackhole, $N-token PD-L1 binder, 31 seeds" --out $A/quality$N.json > $A/quality$N.log 2>&1
echo "quality rc=$? $(tail -1 $A/quality$N.log | cut -c1-200)"
same=0; diff=0
for b in $A/mn$N/pdl1_binder_*_generated.cif; do
  [ -e "$b" ] || continue
  if cmp -s <(grep -E '^(ATOM|HETATM)' $b) <(grep -E '^(ATOM|HETATM)' $A/bh$N/${b##*/}); then same=$((same+1))
  else diff=$((diff+1)); echo "bh ${b##*/}: branch DIFFERS from main"; fi
done
echo "bh-identity $N: $same byte-identical, $diff differ (branch vs main, same chip)"
