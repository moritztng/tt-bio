#!/bin/bash
# Copies the CIFs pvo_session.sh leaves on the box (pvo-accuracy's acc/ arms, RFD4's designs) to pc while the box
# runs, and holds the box's PVO-DONE (box_rfd4_pvo.sh waits on /root/kc/HOLD-PULL) until a sync after RFD4-DONE.
#   setsid nohup bash pull_cifs.sh <label> > results/<label>/pull_cifs.log 2>&1 &
set -u
L=${1:?label}; P=$(cd "$(dirname "$0")" && pwd); OUT=$P/results/$L/cif; mkdir -p $OUT
S=$(cat $P/results/$L/ssh); E="${S% root@*}"; H=root@${S##*root@}
say(){ echo "$(date -u +%FT%TZ) pull_cifs: $*"; }
until timeout 30 $S 'touch /root/kc/HOLD-PULL' 2>/dev/null; do sleep 60; done; say "hold set"
sync1(){ timeout 3600 rsync -a --prune-empty-dirs --include '*/' --include '*.cif' --include '*.cif.gz' --exclude '*' \
  -e "$E" $H:/root/kc/pvo/ $OUT/; }
while :; do
  done_=$(timeout 30 $S 'ls /root/kc/pvo/rfd4/RFD4-DONE /root/kc/pvo/rfd4/RFD4-FAILED /root/kc/pvo/PVO-FAILED 2>/dev/null')
  sync1 && say "synced $(find $OUT -name '*.cif*' | wc -l) cifs"
  [ -n "$done_" ] && sync1 && break
  sleep 300
done
timeout 30 $S 'rm -f /root/kc/HOLD-PULL' && say "hold released, $(find $OUT -name '*.cif*' | wc -l) cifs on pc"
