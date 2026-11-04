#!/bin/bash
# Waits for a session's burn gate to pass, then hands over to pull_cifs.sh. Exits if the session ends first, so it
# never outlives its box.
#   setsid nohup bash pull_cifs_starter.sh <label> > results/<label>/pull_cifs.log 2>&1 &
set -u
L=${1:?label}; P=$(cd "$(dirname "$0")" && pwd); R=$P/results/$L
while :; do
  [ -f $R/SESSION-END ] && { echo "$(date -u +%FT%TZ) session ended before a box passed"; exit 0; }
  [ -f $R/pvo/BURN ] && [ "$(cat $R/pvo/BURN)" = 0 ] && break
  sleep 60
done
echo "$(date -u +%FT%TZ) burn passed, starting pull_cifs"
exec bash $P/pull_cifs.sh $L
