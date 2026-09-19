#!/bin/sh
# Total RSS of each rung's process GROUP. rung.sh samples the parent pid, which is a click
# process that stays at a few hundred MB while the spawned worker does all the work at tens of
# GB, so its column cannot see a stall. Summed over the group, which is what "is this fold
# still growing" actually means.
B=/home/ttuser/.coworker/wt/cov-below-bar-openfold3-bhp150a/perf/of3cov
while :; do
  for d in "$B"/out/of3_*_dev*; do
    p=$(ps -eo pid,args | grep "[t]t_bio.main predict.*$(basename "$d")" | awk "{print \$1}" | head -1)
    [ -n "$p" ] || continue
    tot=$(ps -o rss= -g "$(ps -o pgid= -p "$p" | tr -d " ")" 2>/dev/null | awk "{s+=\$1} END {print s}")
    printf "%s %s\n" "$(date +%s)" "${tot:-NA}" >> "$d/grouprss.log"
  done
  sleep 15
done
