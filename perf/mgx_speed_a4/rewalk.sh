#!/usr/bin/env bash
# Re-walk one chip's model after its first walk: rungs.py skips every rung that already has a quiet
# unit and gives a loud rung up to 1 + EXTRA more units, so this only re-times what a co-tenant spike
# pushed over the 1.5x load ceiling.
#   rewalk.sh <wait_pid> <card> <model> <rungs> [passes=3]
set -u
pid=$1 card=$2 model=$3 rungs=$4 passes=${5:-3}
cd "$(dirname "$0")/../.."
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
for i in $(seq "$passes"); do
  loud=$("$HOME/env/bin/python" - "$model" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from gate_guard import DEFAULT_LOAD_CEILING as C
m = sys.argv[1]
cells = [json.loads(x) for x in Path(f"perf/mgx_speed_a4/runs/{m}.jsonl").read_text().splitlines() if x.strip()]
timed = [c for c in cells if c["tag"] != "warmup" and not c.get("error") and not c.get("contended")]
rungs = {c["rung"] for c in timed}
quiet = {c["rung"] for c in timed if (c.get("load") or {}).get("max", C + 1) <= C}
print(len(rungs - quiet))
PY
)
  [ "$loud" = 0 ] && break
  echo "[$(date -u +%FT%TZ)] REWALK $i $model: $loud rungs without a quiet unit" >> "perf/mgx_speed_a4/logs/$model.log"
  bash perf/mgx_speed_a4/launch.sh "$card" "$model" "$rungs"
done
