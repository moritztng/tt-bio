#!/usr/bin/env bash
# Launch the plan once, the first time the 5-min loadavg reads under 1.0x nproc, then exit.
#   start_when_quiet.sh [hours]      detached, one instance, gives up after [hours] (default 12)
# logs/started records when it fired and the load it saw; once it exists nothing launches again.
cd "$(dirname "$0")/../.."
log=perf/mgx-speed/logs; mkdir -p "$log"
if [ "${1:-}" != --run ]; then
    for p in $(pgrep -f "perf/mgx-speed/start_when_quiet[.]sh --run"); do kill "$p"; done
    setsid nohup bash "$0" --run "${1:-12}" </dev/null >>"$log/start_when_quiet.log" 2>&1 &
    exit 0
fi
end=$(( $(date +%s) + ${2:-12} * 3600 ))
n=$(nproc)
while [ "$(date +%s)" -lt "$end" ]; do
    [ -e "$log/started" ] && exit 0
    read -r _ l5 _ < /proc/loadavg
    if awk -v l="$l5" -v n="$n" 'BEGIN { exit !(l < n) }'; then
        echo "$(date -u +%FT%TZ) load5 $l5 nproc $n $(git rev-parse --short HEAD)" > "$log/started"
        "$HOME/env/bin/python" perf/mgx-speed/plan.py go >> "$log/started"
        exit 0
    fi
    sleep 60
done
echo "$(date -u +%FT%TZ) gave up: load5 never under $n"
