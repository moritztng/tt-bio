#!/usr/bin/env bash
# Pull poses off a rented eval box, destroy it when the dock ends or the cap hits.
#
#   usage: eval_box_watch.sh <label> <cap_h> [deadline_epoch]
#
# Split out of eval_box_session.sh on 2026-11-27. The session script launched the detached dock
# over ssh and then never got that ssh back: the remote `setsid nohup ... >log 2>&1 </dev/null &`
# had already written its pid, but the ssh channel stayed open (sshd on these vast images does not
# close it), so the setup step blocked forever, the monitor loop never started, 55 min of poses
# were never pulled and nothing was watching the cap. Launching the dock and watching it are two
# jobs; a stuck launcher must not stop the pull-back. eval_box_setup.sh now bounds the launch ssh
# and verifies the pid separately, and this watcher survives on its own.
set -u
L=${1:?usage: eval_box_watch.sh <label> <cap_h> [deadline_epoch]} CAP_H=${2:?cap hours}
OUT=${OUT:-$HOME/scratch/dfd/eval}
SLUG=${SLUG:-dfd-eval} S4=${S4:-/home/moritz/.coworker/state/dfd}
. "${VL:-/home/moritz/.coworker/scripts/vast}/lib.sh"
S=$(cat "$OUT/ssh-$L"); read -r I _ < "$OUT/instance-$L"
DEADLINE=${3:-$(( $(date +%s) + CAP_H * 3600 ))}
mkdir -p "$OUT/poses"

pull(){ timeout 600 rsync -a -e "${S% *}" "${S##* }:dfd/poses/" "$OUT/poses/" 2>/dev/null; }
done_n(){ find "$OUT/poses" -name rank1.sdf -size +0 | wc -l; }

END=""
while :; do
  sleep 300
  pull
  timeout 60 $S 'tail -3 ~/dfd/dock.log' > "$OUT/dock-remote-$L.log" 2>/dev/null
  grep -q DOCK_DONE "$OUT/dock-remote-$L.log" 2>/dev/null && END=done
  timeout 30 $S 'kill -0 $(cat ~/dfd/dock.pid)' >/dev/null 2>&1 || [ -n "$END" ] || END="driver gone, $(tail -1 "$OUT/dock-remote-$L.log")"
  [ -n "$END" ] && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && { END="cap $CAP_H h"; break; }
  vl_log "$I $(done_n)/428 docked"
done
pull
vl_log "$I END=$END, $(done_n)/428 complexes have a non-empty rank1.sdf"
vl_destroy "$I" "$END ($(done_n)/428 docked)"
