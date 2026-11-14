#!/bin/bash
# Second census attempt in the slot between the headline --warm 9 run and the session cap. The first census
# (box_bc4.sh) died at 12:48:28Z with CUDA_ERROR_LAUNCH_FAILED inside its profiler window and wrote no trace.
# Waits for the GPU to be empty, runs only if the cap leaves >= 5 minutes, pulls the trace as soon as it exists.
set -u
P=$(cd "$(dirname "$0")" && pwd); OUT=$P/results/bc4-b200; S=$(cat $OUT/ssh)
CAP=${CAP:-$(date -d 2026-10-19T13:27:00Z +%s)}
say(){ echo "$(date -u +%FT%TZ) census-tail: $*"; }
R(){ rsync -az -e "${S% root@*}" "$@"; }; H=root@${S##*root@}
while :; do
  n=$(timeout 30 $S 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' 2>/dev/null | tail -1)
  [ "${n:-1}" = 0 ] && break
  [ $(date +%s) -ge $((CAP-300)) ] && { say "GPU never freed before the cap, no census"; exit 3; }
  sleep 20
done
say "GPU empty, $(( CAP - $(date +%s) )) s to the cap"
[ $(date +%s) -ge $((CAP-300)) ] && { say "too close to the cap, no census"; exit 3; }
R $P/bc4_census.py $H:/root/kc/
timeout 30 $S 'mv /root/kc/bc4/census /root/kc/bc4/census-1 2>/dev/null; cd /root/kc && setsid nohup /opt/bc2-venv/bin/python bc4_census.py --bc2 /root/bc2 --out /root/kc/bc4/census --trace-after 8 --trace-s 4 > /root/kc/bc4/census2.log 2>&1 < /dev/null &'
say "census launched"
while [ $(date +%s) -lt $((CAP-20)) ]; do
  sleep 30; mkdir -p $OUT/census2; R $H:/root/kc/bc4/census/ $OUT/census2/ 2>/dev/null; R $H:/root/kc/bc4/census2.log $OUT/ 2>/dev/null
  [ -f $OUT/census2/census.json ] && break
done
say "final pull: $(find $OUT/census2/trace -type f 2>/dev/null | wc -l) trace files"
