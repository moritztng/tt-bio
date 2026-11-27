#!/bin/bash
# D47 bar needs a quiet host: wait (<= 20 h) until qb2 1-min AND 5-min loadavg are below 3, then run the exchange and
# sharded-step 16-rank legs (run_ex16q.sh, run_opt16q.sh) into ~/r4t/scale/ex16quiet/. Flag line "QUIET" in its load file.
O=~/r4t/scale/ex16quiet; P=$(dirname $(readlink -f $0)); deadline=$(( $(date +%s) + 20*3600 ))
until awk "{exit !(\$1<3 && \$2<3)}" /proc/loadavg; do [ $(date +%s) -gt $deadline ] && { mkdir -p $O; echo "never quiet" > $O/never; exit 3; }; sleep 120; done
mkdir -p $O; $P/run_ex16q.sh $O; $P/run_opt16q.sh $O
