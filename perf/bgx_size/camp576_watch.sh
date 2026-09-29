#!/bin/bash
# Records the 576-axis campaign's outcome when it reaches its stop condition, so the result
# survives the launching turn. Append-only to its OWN file: never rewrites bgx-size.md, because
# a whole-file rewrite by a watcher would drop whatever a concurrent editor had in flight.
OUT=/home/ttuser/.coworker/wt/bgx-size/perf/bgx_size/out/camp544
RES=/home/ttuser/.coworker/state/bgx-size-camp576-result.md
DEADLINE=$(( $(date +%s) + 21600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if [ -f "$OUT/rung.json" ] && ! pgrep -f "out/perf/bgx_size/out/camp544" >/dev/null 2>&1 \
     && ! pgrep -f "camp544" >/dev/null 2>&1; then
    {
      echo "# bgx-size: the 576-axis campaign finished"
      echo
      echo "Recorded by camp576_watch.sh at $(date -u +%Y-%m-%dT%H:%M:%SZ), after the launching"
      echo "turn had ended. hIL2R (2 chains) + 146-residue binder, TRUE Evoformer axis 576 --"
      echo "the largest axis that completes -- card 1 on qb1 p150a, \`--trajectories 1\` explicit."
      echo
      echo '```'
      python3 -c "
import json
d = json.load(open('$OUT/rung.json'))
for k in ('evoformer_axis','complex_residues','target','binder','rounds_completed',
          'seconds_per_round','resident_peak_gb','free_at_peak_gb','aiclk_in_round',
          'trajectories_arg','auto_would_choose','wall_seconds','error'):
    if k in d: print(f'{k:22} {d[k]}')
" 2>&1
      echo '```'
      echo
      echo "## accepted designs"
      echo
      if [ -f "$OUT/3_Ranked/!_Ranked.csv" ]; then
        echo '```'; cat "$OUT/3_Ranked/!_Ranked.csv"; echo '```'
      else
        echo "No \`3_Ranked/!_Ranked.csv\`. Campaign tail follows -- read it before concluding"
        echo "anything: a campaign can stop on its design target, on its trajectory cap, or on"
        echo "an error, and those are three different outcomes."
      fi
      echo
      echo "## campaign log tail"
      echo
      echo '```'; tail -25 "$OUT.log"; echo '```'
    } > "$RES"
    exit 0
  fi
  sleep 120
done
echo "camp576_watch: 6 h deadline passed with the campaign still running or its rung.json absent." > "$RES"
