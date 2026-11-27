#!/usr/bin/env bash
# Collect the per-complex dock refusal reasons into one json map, id -> reason.
#
# pose_eval.py prints "FAILED <id>: <why>" from the worker subprocess, so the reason lands in
# <poses>/logs/dock-*.log on the host that ran the chunk. The tick pools poses with
# --exclude logs, so those reasons never reach pc on their own. Without them a complex with no
# pose is just a hole in the denominator, which is the exact shape this row is here to catch.
#
#   dock_failures.sh > ~/scratch/dfd/eval/failures.json
#   quality_fill.py --failures ~/scratch/dfd/eval/failures.json ...
set -uo pipefail
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
HOSTS="${DFD_DOCK_HOSTS:-anx-glx:/home/ttuser/dfd-eval 172.16.102.112:/home/agent/dfd-eval}"

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
for rec in $HOSTS; do
  IFS=: read -r h root <<<"$rec"
  timeout 120 $SSH "$h" "cat $root/poses/logs/dock-*.log 2>/dev/null" \
    | grep -E '^(FAILED|skip) ' >>"$tmp" 2>/dev/null
done

python3 -c '
import json, re, sys
out = {}
for ln in open(sys.argv[1]):
    ln = ln.rstrip("\n")
    m = re.match(r"^FAILED (\S+): (.*)$", ln) or re.match(r"^skip (\S+): (.*)$", ln)
    if m:
        out.setdefault(m.group(1), m.group(2))
json.dump(out, sys.stdout, indent=1)
print()
' "$tmp"
