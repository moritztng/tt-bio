#!/bin/bash
# A bcp-evo round sitting: arms alternated at the process boundary, ABBAAB, on one card.
#   round_sit.sh [rounds] [out subdir] ["off levers"] ["on levers"]
# Defaults are stage 1's: off = main's program (the gated move's entry not installed), on = the
# entry with its fused backward. A later stage passes its own two lever sets.
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
o=perf/bcp_evo/out/${2:-round}; mkdir -p "$o"
base=tri_att_sdpa_hifi,rne_add
read -ra off <<< "${3:-TT_BIO_TAPED_KERNELS=$base TT_BIO_GATED_BW_FUSED=0 TT_BIO_NOGRAD_INFERENCE=0}"
read -ra on <<< "${4:-TT_BIO_TAPED_KERNELS=$base,reblock_permute_gated TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=0}"
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for tag in off1 on1 on2 off2 off3 on3; do
    if [ "${tag%?}" = on ]; then lv=("${on[@]}"); else lv=("${off[@]}"); fi
    echo "=== $tag (${lv[*]}, $r rounds) $(date -u +%FT%TZ)"
    OUT_DIR=$o perf/bcp_evo/round_arm.sh "$tag" "$r" "${lv[@]}" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
~/bcx_e2e_venv/bin/python3 perf/bcp_device/split.py "$o"/off1 "$o"/on1 "$o"/on2 "$o"/off2 "$o"/off3 "$o"/on3 > "$o/split.txt" 2>&1
