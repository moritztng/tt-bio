#!/bin/bash
# Work one card through a comma-separated list of models, record-then-check each.
# $1 = card, $2 = model[,model...].  Env passed through to drive.sh: MAXLOAD, WAIT_S, CEILING.
#
# Sequential on purpose. Two ladders on one card serialise on the device anyway and would each
# read the other as contention, and the whole point of drive.sh is that a model's record and its
# check see the same box.
#
# Skips a model whose record already landed at this HEAD: _size_ladder_carry_rungs refuses to
# carry across a commit change, so a relaunch after a reset must not start the ladder over for a
# model that finished, and must not half-finish one either.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"
head=$(git rev-parse --short HEAD)
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

IFS=, read -ra models <<< "$2"
for m in "${models[@]}"; do
  have=$(python3 - "$m" "$head" <<'PY'
import json, sys, pathlib
model, head = sys.argv[1], sys.argv[2]
p = pathlib.Path("docs/size_ladder_baseline.d") / f"{model}.json"
try:
    e = json.loads(p.read_text())["cards"]["p300c"]["models"][model]
except Exception:
    print("no"); raise SystemExit
print("yes" if str(e.get("commit", "")).startswith(head[:9]) else "no")
PY
)
  if [ "$have" = "yes" ]; then
    echo "$(stamp) skip $m: already recorded at $head"
    continue
  fi
  echo "$(stamp) === $m on card $card ==="
  bash "$WT/perf/sizeladder_0920/drive.sh" "$card" "$m"
  echo "$(stamp) === $m done rc=$? ==="
done
echo "$(stamp) queue on card $card finished"
