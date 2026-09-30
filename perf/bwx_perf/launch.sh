#!/bin/bash
# Launch one perf/bwx_perf/sit.py sitting on a dev Galaxy chip, detached, costing the box ONE
# agent restart: stop the agent so its worker lets go of the chip, start the sitting, wait until
# the sitting holds the chip's lease, start the agent again (it then serves every other chip and
# leaves this one alone until the sitting releases it).
#   launch.sh <tag> <chip> <sit.py args after --out...>
# Run on the Galaxy from a tt-bio checkout. Needs the box's japanfold env and a python with BC2.
set -uo pipefail
tag=$1; chip=$2; shift 2
root=$(cd "$(dirname "$0")/../.." && pwd)
py=${BWX_PY:-$HOME/bwx/venv/bin/python}
out=${BWX_OUT_ROOT:-$HOME/bwx-perf/out}/$tag
mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }
. ~/japanfold/env.sh
export PYTHONPATH=$root BCX_BC2=${BCX_BC2:-$HOME/bwx/bc2}
export TT_METAL_CACHE=${TT_METAL_CACHE:-$HOME/bwx/cache/tt-metal}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/bwx/cache/xla}
export TT_BIO_LEASE_HOLDER=worker:bwx-perf
say "launch tag=$tag chip=$chip load=$(cut -d' ' -f1-3 /proc/loadavg) commit=$(git -C "$root" rev-parse --short HEAD)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for _ in $(seq 60); do pgrep -f japanfold.chipworker >/dev/null || break; sleep 2; done
setsid nohup "$py" -u "$root/perf/bwx_perf/sit.py" --chip "$chip" --out "$out" "$@" \
    > "$out/sit.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$out/pid"; say "sitting pid $pid"
lease=$(ls ~/japanfold/state/leases/*-card"$chip".json)
for _ in $(seq 120); do
    grep -q "\"pid\": $pid" "$lease" 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
say "lease: $(cat "$lease")"
sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"
