#!/bin/bash
# Verify a list of models against what was just recorded. $1 = card, $2 = model[,model...].
# Env: CEILING (gate preflight multiple of nproc), STALL_S.
#
# Separate from the record queues because the two want different load ceilings. A record queue
# runs with three siblings on the other cards and reads loadavg ~20, which is my own device-bound
# folds sitting in I/O wait rather than CPU starvation -- nesso1 re-recorded under exactly that
# and came back within 0.09 of its previous exponents. But the gate's preflight only sees the
# number, so every check the record queues tried was refused at --load-ceiling 1.0. Running the
# checks as their own pass lets the ceiling reflect whose load it is.
#
# A check is never sliced: the gate refuses --size-ladder-rungs in check mode because "a check
# over a subset of the ladder passes without reading the rungs where a lever most often goes
# dark", and RELEASE_GATE_SIZE_RUNGS would bypass that silently.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"
log="$WT/perf/sizeladder_0920/logs"; work="$WT/perf/sizegate/work"
stall_s="${STALL_S:-300}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

IFS=, read -ra models <<< "$2"
for m in "${models[@]}"; do
  echo "$(stamp) === check $m on card $card, loadavg $(cut -d' ' -f1 /proc/loadavg) ==="
  CEILING="${CEILING:-1.6}" bash "$WT/perf/sizeladder_0920/chk.sh" "$card" "$m" \
    > "$log/chk_${m}_c${card}.log" 2>&1 &
  pid=$!
  # Same stall watchdog as drive2.sh: the 768 aa wedge hits a check exactly as it hits a record.
  ( last=0
    while [ -d "/proc/$pid" ]; do
      sleep 30
      newest=$(ls -t "$work/$m"-*.log 2>/dev/null | head -1); [ -z "$newest" ] && continue
      now=$(stat -c %Y "$newest" 2>/dev/null || echo 0)
      if [ "$now" != "$last" ]; then last=$now; continue; fi
      if [ $(( $(date +%s) - now )) -ge "$stall_s" ]; then
        for p in $(pgrep -f "tt_bio.main predict $work/" 2>/dev/null); do
          echo "$(stamp) STALL on $(basename "$newest") -- SIGKILL fold pid $p"; kill -9 "$p" 2>/dev/null
        done
        last=0
      fi
    done ) & wd=$!
  wait "$pid"; rc=$?
  kill "$wd" 2>/dev/null
  verdict=$(grep -oE "GATE (PASS|FAIL)[^)]*" "$log/chk_${m}_c${card}.log" | tail -1)
  echo "$(stamp) === check $m rc=$rc ${verdict:-no verdict line} ==="
done
echo "$(stamp) check queue on card $card finished"
