#!/bin/bash
# wedge_sweep.sh -- fleet-wide wedge audit for the deep-N galaxy windows (p27-p30).
#
# Lists every fold older than 10 minutes that has NO live spawn_main descendant (the
# multiprocessing device worker). A fold without a worker after 10 min is wedged at
# device-open; a fold WITH one is healthy REGARDLESS of log size or main-pid CPU --
# the pass-212 methodology: main-pid freeze and cold-compile 0-byte logs never
# discriminate; the spawn grandchild's existence + CPU accrual is the health signal.
#
# PROBE HYGIENE (pass-226 lesson, binding): anchor the walk wrapper -> main -> spawn_main
# explicitly. A bare `pgrep -f "predict.*TARGET"` matches the timeout WRAPPER first (its
# cmdline embeds the predict command; lowest pid wins) and reads the wrong process --
# that inverted a healthy fold into a false wedge verdict in pass 226.
#
# Runs on the galaxy as cust-team. Read-only (no kills); conclusion and any action stay
# with the operator (kill-safety: resolve, print, confirm, kill by literal pid).
# usage: bash wedge_sweep.sh
for w in $(pgrep -f "^timeout 21600 env TT_VISIBLE_DEVICES"); do
  chip=$(tr '\0' ' ' < /proc/$w/cmdline | grep -o "TT_VISIBLE_DEVICES=[0-9]*" | grep -o "[0-9]*$")
  m=$(ps -o pid= --ppid $w | head -1 | tr -d ' ')
  [ -z "$m" ] && continue
  et=$(ps -o etimes= -p $m | tr -d ' ')
  [ "$et" -lt 600 ] && continue
  tgt=$(tr '\0' ' ' < /proc/$m/cmdline | grep -o "out_dir [^ ]*" | awk '{print $2}' | xargs basename 2>/dev/null)
  found=0
  for d in $(ps -o pid= --ppid $m); do
    ps -o cmd= -p $d | grep -q spawn_main && found=1
    for dd in $(ps -o pid= --ppid $d 2>/dev/null); do
      ps -o cmd= -p $dd 2>/dev/null | grep -q spawn_main && found=1
    done
  done
  [ $found = 0 ] && echo "WEDGE-SUSPECT chip=$chip wrapper=$w main=$m etimes=${et}s tgt=$tgt"
done
echo sweep-done
