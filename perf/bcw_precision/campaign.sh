#!/bin/bash
# DESIGNS and PERDESIGN for TT_BIO_COTANGENT_B8: two real PD-L1 campaigns on qb2 card 3 through the
# user-facing entry, same budget, binder and seed as bcp-land (perf/bcp_land/campaign.sh), so the
# trajectory ids pair; only the flag differs. Sequential, after the round sitting, on one card.
#   campaign.sh [budget] [wait_pid]
set -uo pipefail
cd "$(dirname "$0")/../.."
budget=${1:-6}; wait_pid=${2:-}
o=perf/bcw_precision/out/campaign; mkdir -p "$o"
if [ -n "$wait_pid" ]; then while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done; fi
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -6 | cut -c1-160; } >> "$o/cotenants.txt";
    sleep 120; done ) &
snap=$!
trap "kill $snap 2>/dev/null" EXIT
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcw_precision/out/xlacache
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcw-precision
for a in off:0 on:1; do
    IFS=: read -r tag val <<< "$a"
    out=$o/$tag; rm -rf "$out"; mkdir -p "$out"
    echo "=== campaign $tag (TT_BIO_COTANGENT_B8=$val, budget=$budget, head $(git rev-parse --short HEAD)) $(date -u +%FT%TZ)"
    TT_BIO_COTANGENT_B8=$val timeout 7200 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
        perf/bcx_p10_campaign/campaign_run.py --trajectories auto --max-trajectories "$budget" \
        --binder 146 --params /home/ttuser/bcx_e2e/af2_params --out "$out" > "$o/$tag.log" 2>&1 \
        || echo "  campaign $tag exited $?"
    echo "=== campaign $tag done $(date -u +%FT%TZ)"
done
echo "=== campaign sitting done $(date -u +%FT%TZ)"
