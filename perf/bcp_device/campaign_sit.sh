#!/bin/bash
# The DESIGNS and TRAJECTORY legs for one lever: two real PD-L1 campaigns on qb1 card 0 through
# the user-facing entry (`perf/bcx_p10_campaign/campaign_run.py`, the shipped auto default, so
# `fast_round` arms every lever), same budget, same seed, same commit; only the flag differs.
# Sequential, because they share the card and a campaign is the thing being timed.
#   campaign_sit.sh FLAG [budget] [wait_pid]
# Waits for wait_pid first (the round sitting) so the two never share the card.
set -uo pipefail
cd "$(dirname "$0")/../.."
flag=$1; budget=${2:-6}; wait_pid=${3:-}
o=perf/bcp_device/out/campaign; mkdir -p "$o"
if [ -n "$wait_pid" ]; then while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done; fi
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -6 | cut -c1-160; } >> "$o/cotenants.txt";
    sleep 120; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_device/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcp-device
for a in off:0 on:1; do
    IFS=: read -r tag val <<< "$a"
    out=$o/$tag; rm -rf "$out"; mkdir -p "$out"
    echo "=== campaign $tag ($flag=$val, budget=$budget) $(date -u +%FT%TZ)"
    env "$flag=$val" timeout 7200 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
        perf/bcx_p10_campaign/campaign_run.py --trajectories auto --max-trajectories "$budget" \
        --binder 146 --params /home/ttuser/bcx_e2e/af2_params --out "$out" > "$o/$tag.log" 2>&1 \
        || echo "  campaign $tag exited $?"
    echo "=== campaign $tag done $(date -u +%FT%TZ)"
done
echo "=== campaign sitting done $(date -u +%FT%TZ)"
