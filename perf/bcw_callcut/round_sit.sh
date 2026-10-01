#!/bin/bash
# bcw-callcut round sitting: off = ReLU backward composed + OPM rows summed (C1 and C4 off), on =
# both on; every other fast_round lever on in both arms, ABBAAB at the process boundary, one card. AICLK of that card sampled every 2 s.
#   round_sit.sh <card> [rounds] [out subdir]
set -uo pipefail
cd "$(dirname "$0")/../.."
card=$1; r=${2:-9}
o=perf/bcw_callcut/out/${3:-round_stack}; mkdir -p "$o"
( while :; do echo "$(date -u +%s) $(cat /sys/class/tenstorrent/tenstorrent!$card/tt_aiclk) $(cut -d' ' -f1 /proc/loadavg)"; sleep 2; done ) > "$o/aiclk.txt" &
ck=$!
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap $ck 2>/dev/null' EXIT
sed 's/TT_BIO_LEASE_HOLDER=worker:bcp-evo/TT_BIO_LEASE_HOLDER=worker:bcw-callcut/' perf/bcp_evo/round_arm.sh > perf/bcw_callcut/round_arm.sh
chmod +x perf/bcw_callcut/round_arm.sh
for tag in off1 on1 on2 off2 off3 on3; do
    g=$([ "${tag%?}" = on ] && echo 1 || echo 0)
    echo "=== $tag (BCW_RELU_GATED=$g, $r rounds) $(date -u +%FT%TZ)"
    TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card BCW_RELU_GATED=$g TT_BIO_AF2_OPM_ROWS_IN_K=$g BCP_DUO=perf/bcw_callcut/duo_relu.py \
        OUT_DIR=$o perf/bcw_callcut/round_arm.sh "$tag" "$r" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
~/bcx_e2e_venv/bin/python3 perf/bcp_device/split.py "$o"/off1 "$o"/on1 "$o"/on2 "$o"/off2 "$o"/off3 "$o"/on3 > "$o/split.txt" 2>&1
