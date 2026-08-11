#!/bin/bash
# Artifact-based liveness for the opendde-abag 9j4c chunk folds. Run it ON the Galaxy
# (UF-EV-A13-GWH02, user cust-team); it prints one TSV line per chunk and exits non-zero if any
# chunk is stalled or dead, so a caller can poll it instead of guessing from a wall clock.
#
# Why this exists, from two failures in this campaign:
#
#   * A wall-clock DEFER is not a feedback loop. Pass 14 predicted "first verdict at 21:45Z" and
#     nothing checked the folds until then, so an early death would have been invisible for an
#     hour. Liveness has to be polled from artifacts, not from an ETA.
#   * A log line is not liveness and neither is CPU time. Chunk 1 sat inside a hung ttnn.chunk at
#     100 pct CPU with a rising CPU-time counter for 97 minutes, indistinguishable from working by
#     every CPU measure. What separates the two cases is whether the log line ADVANCES, and
#     whether a process and a device handle still exist. All three are checked here.
#
# Exit: 0 all live-or-finished, 1 something is stalled or dead without output, 2 nothing found.
set -u
H=${H:-$HOME/mthuening}
P=$H/p34d
STALL=${STALL:-1500}            # 25 min with no new log line; recycles run 8.9-10.4 min co-tenanted
WANT_CIFS=${WANT_CIFS:-64}
NOW=$(date +%s)

# Which chips a live fold is actually pinned to. Do NOT read this off /dev/tenstorrent: a
# `tt-bio serve` holds an idle worker on all 32 chips, so every chip looks busy there.
declare -A CHIP_PID
while read -r pid; do
  [ -r /proc/$pid/environ ] || continue
  d=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^TT_VISIBLE_DEVICES=//p' | head -1)
  [ -n "${d:-}" ] && CHIP_PID[$d]=$pid
done < <(pgrep -f 'tt_bio.main predict' 2>/dev/null)

bad=0; seen=0
printf 'chunk\tchip\tpid\tlast_line\tage_s\tcifs\tverdict\n'
for dir in $P/odcamp/9j4c_c* $P/od9j4c/9j4c_c*; do
  [ -d "$dir" ] || continue
  c=$(basename "$dir")
  # A retired output dir is not a chunk. od9j4c_r2_fleet.sh renames the dead mps=5 output to
  # `<chunk>.stale.<epoch>` before it re-folds, so the glob above matches one extra directory per
  # retried chunk. Counted as a chunk it has no process and no cifs, scores DEAD_NO_OUTPUT, and
  # pins the exit code to 1 forever — the script can then never report a healthy fleet.
  case "$c" in *.stale.*) continue ;; esac
  base=$(dirname "$dir")
  seen=$((seen+1))

  # Match `<chunk>_mps2.log` AND `<chunk>_r2_mps2.log`: the r2 runner puts its own tag between the
  # chunk name and the mps rung. The old `${c}_mps*` glob missed every r2 log, fell back to the
  # retired mps=5 log, and read its hours-old mtime as a stall on a chunk that was folding fine.
  log=$(ls -t "$base/${c}"*_mps*.log 2>/dev/null | head -1)
  line='-'; age=-1
  if [ -n "${log:-}" ]; then
    line=$(grep -E 'trunk [0-9]+/|diffusion|sample|Done:' "$log" 2>/dev/null | tail -1 | sed 's/^[0-9:]*  *//;s/\[[^]]*\] *//' | cut -c1-40)
    age=$(( NOW - $(stat -c %Y "$log") ))
  fi

  cifs=$(ls "$dir"/opendde_results_9j4c/structures/*.cif 2>/dev/null | wc -l)

  # The chip this chunk is pinned to, and whether that fold's process still exists.
  chip='-'; pid='-'
  for ch in "${!CHIP_PID[@]}"; do
    p=${CHIP_PID[$ch]}
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qF -- "$dir "; then
      chip=$ch; pid=$p; break
    fi
  done

  if [ "$cifs" -ge "$WANT_CIFS" ]; then
    v=DONE
  elif [ "$pid" != '-' ]; then
    if [ "$age" -ge "$STALL" ]; then v=STALL; bad=$((bad+1)); else v=RUNNING; fi
  elif [ -n "${log:-}" ] && grep -q 'Out of Memory' "$log" 2>/dev/null; then
    v=OOM; bad=$((bad+1))
  else
    v=DEAD_NO_OUTPUT; bad=$((bad+1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$chip" "$pid" "$line" "$age" "$cifs" "$v"
done

[ "$seen" -eq 0 ] && exit 2
[ "$bad" -gt 0 ] && exit 1
exit 0
