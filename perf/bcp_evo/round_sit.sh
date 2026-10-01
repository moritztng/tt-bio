#!/bin/bash
# The gated move's round sitting: arms alternated at the process boundary, ABBAAB, on one card.
#   round_sit.sh [rounds]
# off = main's program (the entry not installed); on = the entry with its fused backward.
set -uo pipefail
cd "$(dirname "$0")/../.."
r=${1:-9}
o=perf/bcp_evo/out/round; mkdir -p "$o"
base=tri_att_sdpa_hifi,rne_add
off=("TT_BIO_TAPED_KERNELS=$base" "TT_BIO_GATED_BW_FUSED=0")
on=("TT_BIO_TAPED_KERNELS=$base,reblock_permute_gated" "TT_BIO_GATED_BW_FUSED=1")
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-200; } >> "$o/cotenants.txt";
    sleep 60; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
for tag in off1 on1 on2 off2 off3 on3; do
    if [ "${tag%?}" = on ]; then lv=("${on[@]}"); else lv=("${off[@]}"); fi
    echo "=== $tag (${lv[*]}, $r rounds) $(date -u +%FT%TZ)"
    perf/bcp_evo/round_arm.sh "$tag" "$r" "${lv[@]}" > "$o/$tag.log" 2>&1 || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
