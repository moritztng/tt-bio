#!/bin/bash
# Runs on qb2: grades the Mol-1 DockQ precision A/B (prec_arm.sh) on its CASES.json subset. Arms: gpu, gpu_self, gpu_fp32,
# tt_bh = the default BH arm on the same tree, tt_bh_p = the lever-off arm (pulled from qb1, finished folds only, as
# ab_grade.sh does). The subset's full case records come from ~/mol1-quality/CASES.json. Re-runnable, grading is cached.
#   bash prec_grade.sh NAME DEFAULT_DIR P_DIR SUBSET.json   -> ~/mol1-quality/NAME/{QUALITY.json,work/rows.json}, GRADE-EXIT
set -uo pipefail
cd ~/mol1-quality; N=${1:?name}; A=${2:?default dir}; B=${3:?P dir}; SUB=$(realpath "${4:?subset}"); mkdir -p "$N" "$B"
python3 - "$SUB" "$N/CASES.json" <<'PY'
import json, sys
want = {c["id"] for c in json.load(open(sys.argv[1]))["paired"]}
full = json.load(open("CASES.json"))
full["paired"] = [c for c in full["paired"] if c["id"] in want]
full["headline"] = []
assert len(full["paired"]) == len(want), sorted(want - {c["id"] for c in full["paired"]})
json.dump(full, open(sys.argv[2], "w"), indent=1)
PY
Q=ttuser@tt-quietbox
timeout 120 ssh -o BatchMode=yes $Q "cd ~/mol1-quality/$B && find .runs -name rc | xargs grep -l rc=0" |
while read -r rc; do
  d=${rc%/rc}; s=${d##*/}; c=${d#.runs/}; c=${c%/*}; r=mol1_results_${c%%.json*}/$c/$s
  [ -d "$B/$r" ] || timeout 1800 rsync -a --exclude claim $Q:mol1-quality/$B/$r/ "$B/$r/" --mkpath < /dev/null
done
for f in aiclk.log LEVERS ENGINE_SHA; do rsync -a $Q:mol1-quality/$B/$f "$B/$f"; done
P=$HOME/.coworker/wt/mol1-quality/perf/mol1_quality; G=$P/grade.py
{ PATH=$PWD/env/fb/bin:$PATH nice -n 19 env/fb/bin/python "$G" --inputs inputs --cases "$N/CASES.json" --work "$N/work" \
    --out "$N/QUALITY.json" --arm gpu=arms/gpu --arm gpu_self=arms/gpu_self --arm gpu_fp32=arms/gpu_fp32 \
    --arm tt_bh="$A" --arm tt_bh_p="$B" --ab tt_bh_p=tt_bh --ref-split "$P/ref_split.json"
  echo "GRADE-EXIT rc=$?"; } > "$N/grade.log" 2>&1
