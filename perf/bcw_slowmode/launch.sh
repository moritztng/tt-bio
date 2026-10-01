#!/bin/bash
# Launch perf/bcw_slowmode/chain.py on one chip of a dev
# Wormhole Galaxy, detached, costing the box ONE agent restart for the whole sitting: stop the
# agent so its worker lets go of the chip, start the sitting, wait until it holds the chip's
# lease, start the agent again. The agent then serves every other chip and leaves this one alone
# until the sitting releases it.
#   launch.sh <tag> <chip> <chain.py args after --out>
# Run ON the Galaxy from a tt-bio checkout. Read /v1/cluster first: a restart removes all ~32 of
# the box's chips from the pool for the restart window, which is harmless at 0 running jobs and
# not otherwise (state/bwx/CHIPS.md, the rule added 2026-09-29).
set -uo pipefail
tag=$1; chip=$2; shift 2
root=$(cd "$(dirname "$0")/../.." && pwd)
py=${B2P_PY:-$HOME/bwx/venv/bin/python}
script=chain
out=${B2P_OUT_ROOT:-$HOME/bcw-slowmode/out}/$tag
mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }
. ~/japanfold/env.sh
export BCX_BC2=${BCX_BC2:-$HOME/bwx/bc2}
export PYTHONPATH=$root:$BCX_BC2
export TT_METAL_CACHE=${TT_METAL_CACHE:-$HOME/bwx/cache/tt-metal}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/bwx/cache/xla}
export TT_BIO_LEASE_HOLDER=worker:bcw-slowmode
say "launch tag=$tag chip=$chip load=$(cut -d' ' -f1-3 /proc/loadavg) commit=$(git -C "$root" rev-parse --short HEAD)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for _ in $(seq 60); do pgrep -f japanfold.chipworker >/dev/null || break; sleep 2; done
setsid nohup "$py" -u "$root/perf/bcw_slowmode/chain.py" --chip "$chip" --out "$out" "$@" \
    > "$out/$script.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$out/pid"; say "$script pid $pid"
lease=$(ls ~/japanfold/state/leases/*-card"$chip".json 2>/dev/null)
for _ in $(seq 180); do
    grep -q "\"pid\": $pid" "$lease" 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
say "lease: $(cat "$lease" 2>/dev/null)"
sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"
