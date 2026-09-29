#!/bin/bash
# One handover for the whole redo sitting. A handover costs the BOX its place in /v1/cluster for
# the restart window, not one chip (root-caused 2026-09-29: bwx-serve read 64 chips / 2 agents and
# could not explain it), so the sitting is batched behind a single stop/start and the caller must
# have checked jobs.running is 0 first.
#
# Waits on the lease HOLDER string rather than a child pid: the sitting runs many python processes
# under one bash, so job.sh's pid match would never fire and would hold the agent down for its
# full 300 s cap.
set -uo pipefail
cd ~/bwx
CHIP=${BWX_CHIP:-30}
out=~/bwx/out/sitting2; mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }
. ~/japanfold/env.sh
export TT_METAL_CACHE=$HOME/bwx/cache/tt-metal PYTHONPATH=$HOME/bwx/tt-bio BCX_BC2=$HOME/bwx/bc2
export JAX_COMPILATION_CACHE_DIR=$HOME/bwx/cache/xla
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP
export TT_BIO_LEASE_HOLDER=worker:bwx-bringup TT_BIO_LEASE_TIMEOUT=28800
export BWX_OUT=$out BWX_CHIP=$CHIP
say "sitting2 launch chip=$CHIP load=$(cat /proc/loadavg | cut -d' ' -f1-3)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 60); do pgrep -f "japanfold.chipworker" >/dev/null || break; sleep 2; done
setsid nohup bash ~/bwx/sitting2.sh < /dev/null > "$out/job.log" 2>&1 &
pid=$!; echo "$pid" > "$out/pid"; say "sitting pid $pid"
setsid nohup ~/bwx/venv/bin/python -u $HOME/bwx/aiclk.py "$pid" "$out/aiclk.jsonl" < /dev/null > "$out/aiclk.log" 2>&1 &
for i in $(seq 90); do
  grep -q "worker:bwx-bringup" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
say "lease on card $CHIP: $(grep -o "\"holder\": \"[^\"]*\"" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null | tail -1)"
sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"
say "sitting2 detached; log $out/job.log, stage markers $out/DONE-*"
