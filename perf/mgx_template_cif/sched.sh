#!/bin/bash
# Fold the template cells of the given models on at most two whglx chips at a time, taking a
# chip only when its lease file is absent or released (>90 s ago) and its flock is free. A model whose
# batch lost the chip to a co-tenant at open (DeviceInUseError) goes back on the queue.
#   sched.sh <model> ...   (logs to perf/mgx_template_cif/sched.log)

cd "$(dirname "$0")/../.."
L=$HOME/leases; LOG=perf/mgx_template_cif/sched.log
Q=("$@"); declare -A RUN=()   # card -> "pid model"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
free_card() {
  for c in $(seq 0 31); do
    case " 1 24 25 26 27 " in *" $c "*) continue;; esac
    [ -n "${RUN[$c]:-}" ] && continue
    f=$L/j10glx02-card$c.json
    # Released more than 90 s ago: a row that runs batches back to back releases for a moment
    # between them and re-takes the chip at once.
    if [ -f "$f" ] && ! python3 -c "import json,sys,time; r=json.load(open(sys.argv[1]))['released']; sys.exit(0 if r and time.time()-r>90 else 1)" "$f" 2>/dev/null; then continue; fi
    flock -n "$f" true 2>/dev/null || continue
    echo "$c"; return 0
  done
  return 1
}
while [ ${#Q[@]} -gt 0 ] || [ ${#RUN[@]} -gt 0 ]; do
  for c in "${!RUN[@]}"; do
    set -- ${RUN[$c]}; pid=$1 m=$2
    kill -0 "$pid" 2>/dev/null && continue
    unset "RUN[$c]"
    if grep -q "is in use by\|DeviceInUseError" "perf/mgx_matrix/out/$m/run.log" 2>/dev/null; then
      log "requeue $m (card $c contended)"; Q+=("$m")
    else
      log "done $m card $c: $(tail -1 perf/mgx_matrix/out/$m/run.log)"
    fi
  done
  while [ ${#RUN[@]} -lt 2 ] && [ ${#Q[@]} -gt 0 ] && c=$(free_card); do
    m=${Q[0]}; Q=("${Q[@]:1}")
    setsid perf/mgx_template_cif/run.sh "$c" "$m" < /dev/null > /dev/null 2>&1 &
    RUN[$c]="$! $m"; log "start $m card $c pid $!"
    sleep 5
  done
  sleep 20
done
log "all done"
