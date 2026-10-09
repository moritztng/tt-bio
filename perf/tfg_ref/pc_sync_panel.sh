#!/usr/bin/env bash
# pc side: wait for the MSA shards, check depths, push the panel to the box and mark it READY.
# usage: pc_sync_panel.sh PANEL_DIR SSH_PORT SSH_HOST "SHARD_PIDS"
set -euo pipefail
PANEL=$1; PORT=$2; HOST=$3
for p in $4; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
python3 - "$PANEL" <<'PY'
import json, glob, sys
d = {}
for f in glob.glob(f"{sys.argv[1]}/msa_depths.*.json"):
    d.update(json.load(open(f)))
shallow = {t: v for t, v in d.items() if min(v) <= 1}
missing = [t for t in json.load(open(f"{sys.argv[1]}/panel.json"))["targets"]
           if "unpairedMsaPath" not in open(f"{sys.argv[1]}/{t}/{t}_unconstrained.json").read()]
print("targets with MSA", len(d), "shallow", shallow, "missing", missing)
PY
rsync -az --exclude _cif -e "ssh -p $PORT -o BatchMode=yes" "$PANEL/" "root@$HOST:/root/panel/"
ssh -p "$PORT" -o BatchMode=yes "root@$HOST" touch /root/panel/READY
echo SYNCED
