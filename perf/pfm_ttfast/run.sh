#!/bin/bash
# pfm-ttfast sitting on .107 (UF-EV-A4-GWH01): N chips through tt-bio's own lease, the jgp-ours way.
# Stop the agent (its workers close their chips), start one bench process per chip, wait until each holds
# its lease, start the agent again (it serves every other chip and skips the leased ones). Each process
# keeps its device open for all its arms, so the chip goes back exactly once, when the process exits.
# The EXIT trap restarts the agent whatever happens to this script.
cd ~/pfm-ttfast; . ~/japanfold/env.sh
export PATH="$PATH:$HOME/japanfold/msa/venv/bin" TT_BIO_LEASE_HOLDER=pfm-ttfast TT_BIO_LEASE_TIMEOUT=600
RUN=${1:?run dir}; shift
say(){ echo "$(date -u +%FT%TZ) $*"; }
agent_up(){ systemctl is-active -q japanfold-agent@ubuntu || { sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"; }; }
trap agent_up EXIT
mkdir -p "$RUN"
say "start engine $(git -C ~/japanfold/engine rev-parse HEAD) load $(cat /proc/loadavg)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 90); do pgrep -f "japanfold.chipworker" >/dev/null || break; sleep 2; done
say "chipworkers left: $(pgrep -fc japanfold.chipworker)"
PIDS=()
while [ $# -gt 0 ]; do
  CHIP=$1; PLAN=$2; shift 2
  D="$RUN/c$CHIP"; mkdir -p "$D"; ln -sfn ~/pfm-ttfast/msa "$D/msa"
  nohup python bench_wh.py "$D" "$CHIP" 31 complex730.yaml "$PLAN" > "$D.log" 2>&1 &
  PIDS+=($!); say "chip $CHIP pid $! plan $PLAN"
done
for D in "$RUN"/c*/; do
  for i in $(seq 300); do grep -q '"nodes_open"' "$D/bench.jsonl" 2>/dev/null && break; sleep 2; done
  say "$D: $(grep -o '"ev": "nodes_open", "nodes": \[[0-9, ]*\]' "$D/bench.jsonl")"
done
agent_up
for p in "${PIDS[@]}"; do wait $p; say "pid $p rc=$?"; done
say "end load $(cat /proc/loadavg)"
