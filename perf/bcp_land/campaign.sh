#!/bin/bash
# PERDESIGN on qb2: one real PD-L1 campaign on the composed no-flags default, the same entry,
# budget, binder and seed as bcp-device's qb1 campaign (perf/bcp_device/campaign_sit.sh), so the
# trajectory ids pair across the two boards. No TT_BIO_* is set: this is what a user runs.
#   campaign.sh [budget] [wait_pid]     (OUT_DIR, TT_BIO_LEASE_HOLDER and the card env override)
set -uo pipefail
cd "$(dirname "$0")/../.."
budget=${1:-6}; wait_pid=${2:-}
o=${OUT_DIR:-perf/bcp_land/out/campaign}; mkdir -p "$o"
if [ -n "$wait_pid" ]; then while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done; fi
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -6 | cut -c1-160; } >> "$o/cotenants.txt";
    sleep 120; done ) &
snap=$!
trap 'kill $snap 2>/dev/null' EXIT
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$PWD/perf/bcp_land/out/xlacache}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-1} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-1}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcp-land}
out=$o/on; rm -rf "$out"; mkdir -p "$out"
echo "=== campaign on (no flags, budget=$budget, card $TT_VISIBLE_DEVICES, head $(git rev-parse --short HEAD)) $(date -u +%FT%TZ)"
timeout 7200 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_campaign/campaign_run.py \
    --trajectories auto --max-trajectories "$budget" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" > "$o/on.log" 2>&1 || echo "  campaign exited $?"
echo "=== campaign done $(date -u +%FT%TZ)"
