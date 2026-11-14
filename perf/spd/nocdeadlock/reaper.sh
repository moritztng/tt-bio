#!/bin/bash
# reaper.sh: a repro leg whose hang was captured (hangcap.log line rp_<tag> c<N>) is ended with SIGTERM 30 s later,
# so phase3.sh moves to its next arm instead of waiting out the 3600 s timeout. Exits when phase3.sh is gone.
cd ~/spd/spd-nocdeadlock
while pgrep -f phase3.sh > /dev/null; do
  grep -o "hangcap c[23] rp_[A-Za-z0-9_]*" hangcap.log | sort -u | while read _ c tag; do
    m=~/spd/spd-nocdeadlock/.reaped_${tag}_$c; [ -e $m ] && continue
    p=$(pgrep -f "TT_VISIBLE_DEVICES=${c#c} .*" -n 2>/dev/null)
    for q in $(pgrep -f "paircond_repro.py"); do
      grep -qz "TT_VISIBLE_DEVICES=${c#c}" /proc/$q/environ 2>/dev/null || continue
      grep -qz "TT_METAL_LOGS_PATH=.*insp_${tag}_$c" /proc/$q/environ 2>/dev/null || continue
      sleep 30; kill -TERM $q; echo "reaped $tag $c pid $q $(date -u +%FT%TZ)" >> reaper.log
    done; touch $m
  done; sleep 15
done
