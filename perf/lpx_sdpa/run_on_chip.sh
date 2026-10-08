#!/bin/bash
# Take ONE .107 Wormhole chip through tt-bio's own lease, the pfm-ttfast way: refuse if any worker is running a
# job, stop the agent only until our process holds its lease (~10 s), restart it (the EXIT trap restarts it
# whatever happens). usage: run_on_chip.sh CHIP OUTDIR script.py [args...]
cd ~/lpx-sdpa; . ~/japanfold/env.sh
export PATH="$PATH:$HOME/japanfold/msa/venv/bin" TT_BIO_LEASE_HOLDER=lpx-sdpa TT_BIO_LEASE_TIMEOUT=600
CHIP=$1; OUT=$2; shift 2
# LPX_ENGINE: a tt_bio checkout to run instead of the installed engine (our branch, for kernel changes)
[ -n "$LPX_ENGINE" ] && export PYTHONPATH="$LPX_ENGINE"
say(){ echo "$(date -u +%FT%TZ) $*"; }
busy=$(curl -s -m5 localhost:8767/cluster | python -c 'import json,sys; d=json.load(sys.stdin); print(sum(len(w.get("running") or []) for w in d["workers"]))')
[ "$busy" = "0" ] || { say "REFUSED: $busy job(s) running on .107 workers"; exit 3; }
agent_up(){ systemctl is-active -q japanfold-agent@ubuntu || { sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"; }; }
trap agent_up EXIT
mkdir -p "$OUT"
say "engine $(git -C ~/japanfold/engine rev-parse --short HEAD) load $(cat /proc/loadavg)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 90); do pgrep -f "japanfold.chipworker" >/dev/null || break; sleep 2; done
nohup python "$@" > "$OUT/run.log" 2>&1 &
PID=$!; say "chip $CHIP pid $PID"; echo $PID > "$OUT/pid"
for i in $(seq 300); do grep -q '"nodes_open"' "$OUT/run.log" 2>/dev/null && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
say "$(grep -o '"ev": "nodes_open", "nodes": \[[0-9, ]*\]' "$OUT/run.log")"
agent_up
wait $PID; say "pid $PID rc=$?"; say "end load $(cat /proc/loadavg)"
