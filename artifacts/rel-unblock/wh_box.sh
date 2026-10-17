#!/bin/bash
# Get the release gate's WH legs a Galaxy: ask glx_elastic for a maintenance slot on .107 (granted after 15 min
# with JapanFold quiet and the box out), then lend the box to the SPD pool the gate queues into. From then on the
# controller's normal give-back returns it the moment JapanFold queues.
set -u
E="python3 $HOME/.coworker/scripts/glx_elastic.py"
IP=${1:-107}
echo "$(date -u +%FT%TZ) maint ask .$IP for rel-train"
$E maint "$IP" --holder rel-train --wait 14400; rc=$?
echo "$(date -u +%FT%TZ) maint rc $rc"
[ $rc -eq 0 ] || exit $rc
rm -f "$HOME/.coworker/state/glx/requests/maint-$IP.json"
$E act lend "$IP" spd 3 "rel-train: WH legs of release gate 5e250928a (maint slot); give-back rules apply"
echo "$(date -u +%FT%TZ) lend done:"; $E status | grep "^\.$IP"
