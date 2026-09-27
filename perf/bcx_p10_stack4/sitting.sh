#!/bin/bash
# bcx-p10-stack4: TT_BIO_TAPED_CHANNEL_MOVE and TT_BIO_GRAD_FANIN_L1 priced as a pair on the
# composed round. Two four-process passes of perf/bcx_p10_l1fuse/ab.sh, `off on on off` then the
# reverse, so each arm gets 32 timed rounds and a drift over the sitting lands on both arms.
# A co-tenant snapshot every 60 s names who else was on the box while it ran.
set -uo pipefail
cd "$(dirname "$0")/../.."
out=perf/bcx_p10_stack4/out; mkdir -p "$out"
export LEVER="TT_BIO_TAPED_CHANNEL_MOVE TT_BIO_GRAD_FANIN_L1"
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-1} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-1}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack4
( while :; do echo "--- $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"
    ps -eo pid,pcpu,etime,args --sort=-pcpu | awk "NR>1 && \$2>20" | cut -c1-200
    sleep 60; done ) > "$out/cotenants.txt" 2>&1 &
snap=$!
trap "kill $snap" EXIT
bash perf/bcx_p10_l1fuse/ab.sh 9 s4a off on on off
bash perf/bcx_p10_l1fuse/ab.sh 9 s4b on off off on
