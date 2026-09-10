#!/bin/bash
# Pass 2: best-of-N fresh-process draws for every leg pass 1 could not separate from host noise.
# Waits out any process still holding card 1 (node 2) before the first draw.
S=/home/ttuser/.coworker/state/perf-excellence-p150a
for i in $(seq 1 120); do
  h=$(lsof -t /dev/tenstorrent/2 2>/dev/null | head -1)
  [ -z $h ] && break
  sleep 15
done
echo "PASS2-START $(date -u +%FT%TZ) loadavg=$(cut -d' ' -f1 </proc/loadavg)" >> "$S/perf.log"
for spec in "$@"; do
  m=${spec%%:*}; n=${spec##*:}
  "$S/draws.sh" "$m" "$n" pass2
done
echo "PASS2-DONE $(date -u +%FT%TZ)" >> "$S/perf.log"
