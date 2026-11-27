#!/bin/bash
# After a reference run: append the examples that cover any loss term left unexercised, then
# prove every dump replays, then emit the grader's reference half.
#   GEN_PID=<pid> perf/r4t/cover_terms.sh <ref_dir>
# Covering ids come from term_coverage.py: of upstream's 16 test_dataset crops, one carries a
# chiral centre and one carries unindexed tokens, so a set taken in dataset order grades
# signed_volume and insertion_prediction trivially at zero on both sides.
set -u
WT=/home/moritz/.coworker/wt/r4t-accuracy
REF=${1:?ref dir}
PY=/home/moritz/rfd4-cpu-venv/bin/python
cd "$WT" || exit 1
export RFD4_REPO=/home/moritz/rfd4-upstream/RFD4-Proteina \
       DATA_PATH=/home/moritz/r4t/objective/data \
       HF_HOME=/home/moritz/r4t/gpu-baseline/hf OMP_NUM_THREADS=10

while [ -n "${GEN_PID:-}" ] && kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done

MISSING=$(python3 - "$REF" <<'PY'
import json, sys
from pathlib import Path
s = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
have = {k for b in s["batches"] for k, v in b["terms"].items() if v != 0.0}
seen = {b["example_id"] for b in s["batches"]}
cover = {"signed_volume": "2plh__1__1.A__1.G", "insertion_prediction": "7rcu__4__1.U_1.V__1.FA"}
need = sorted({eid for term, eid in cover.items() if term not in have and eid not in seen})
print(",".join(need))
print(len(s["batches"]), file=sys.stderr)
PY
)
N=$(python3 -c "import json,sys;print(len(json.load(open('$REF/summary.json'))['batches']))")
echo "already $N batches; covering ids: ${MISSING:-none}"

if [ -n "$MISSING" ]; then
  COUNT=$(awk -F, '{print NF}' <<<"$MISSING")
  "$PY" perf/r4t/loss_parity_reference.py --out "$REF" --ids "$MISSING" \
        --n "$COUNT" --index-offset "$N" || echo "COVER_RC=$?"
fi

"$PY" perf/r4t/loss_parity_reference.py --replay --out "$REF"; echo "REPLAY_RC=$?"
python3 perf/r4t/loss_parity_reference.py --out "$REF" --emit-ref "$REF/../ref_half.json"
