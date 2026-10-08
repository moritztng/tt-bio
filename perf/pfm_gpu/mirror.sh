#!/bin/bash
# Mirrors a live session's results to pc every 5 min, so a host that stops the container mid-run (54728379,
# 54738078) does not take finished arms with it. Runs beside session.sh; reads the box from results/<label>/ssh.
#   usage: mirror.sh <label>   (exits when /tmp/pfmgpu/<label>.log shows SESSION-END)
L=$1; P=$(cd "$(dirname "$0")" && pwd); M=$P/results/mirror-$L
while ! grep -q SESSION-END /tmp/pfmgpu/$L.log; do
  S=$(cat $P/results/$L/ssh 2>/dev/null); I=$(cut -d' ' -f1 $P/results/$L/instance 2>/dev/null)
  if [ -n "$S" ] && [ -n "$I" ]; then
    mkdir -p $M/$I/acc; E="${S% root@*}"; H=${S##* }
    timeout 240 rsync -a -e "$E" $H:/root/pfm/results/ $H:/root/pfm/setup.log $H:/root/pfm/arms.log $M/$I/ 2>/dev/null
    timeout 240 rsync -a -e "$E" $H:/root/pfm/acc/out/ $M/$I/acc/ 2>/dev/null
  fi
  sleep 300
done
