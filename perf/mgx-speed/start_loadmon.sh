#!/usr/bin/env bash
# (Re)start the box load logger detached, one instance: start_loadmon.sh [minutes]
cd "$(dirname "$0")/../.."
for p in $(pgrep -f "python perf/mgx-speed/loadmon[.]py"); do kill "$p"; done
mkdir -p perf/mgx-speed/logs
setsid nohup "$HOME/env/bin/python" perf/mgx-speed/loadmon.py perf/mgx-speed/logs/load.jsonl "${1:-1440}" \
    </dev/null >/dev/null 2>&1 &
