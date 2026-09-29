#!/bin/bash
# Does BindCraft 2's teardown abort (exit 134, CHIP_IN_USE) harm the next process on the chip?
#   trial.sh <card> <abort|control> <tag>
# abort:   a short real campaign through bindcraft2.run_campaign (perf/bcx_p10_campaign/
#          campaign_run.py, the default trajectories_per_card), then the gate's protenix-v1 +
#          opendde fold legs as the very next process on the card.
# long:    the same, with BindCraft 2's own stage lengths (no step overrides).
# control: the same fold legs with no campaign before them.
# TT_BIO_DEBUG_STDERR keeps tt-bio's stderr filter out of the campaign: its forwarding child dies
# with the parent, so an abort's own message never reaches the log.
# The card's AICLK is sampled once a second for the whole fold.
set -uo pipefail
cd "$(dirname "$0")/../.."
card=$1; kind=$2; tag=$3
o=perf/bc2_teardown/out/$tag; rm -rf "$o"; mkdir -p "$o"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:bc2-teardown-next-job
PY=/home/ttuser/bcx_e2e_venv/bin/python3
echo "trial=$tag card=$card kind=$kind commit=$(git rev-parse --short HEAD) start=$(date -u +%FT%TZ)" > "$o/summary.txt"
steps="--set screen_steps=4 --set refine_steps=2 --set anneal_steps=2 --set harden_steps=1 --set mutate_steps=1"
[ "$kind" = long ] && steps=""
if [ "$kind" != control ]; then
    TT_BIO_DEBUG_STDERR=1 BCX_BC2=/home/ttuser/bcx_e2e/bc2 JAX_COMPILATION_CACHE_DIR=$PWD/perf/bc2_teardown/out/xlacache \
    timeout 5400 $PY -u perf/bcx_p10_campaign/campaign_run.py --max-trajectories 3 --binder 146 \
        --params /home/ttuser/bcx_e2e/af2_params --out "$o/campaign" $steps > "$o/campaign.log" 2>&1
    rc=$?
    echo "campaign_rc=$rc chip_in_use=$(grep -c CHIP_IN_USE "$o/campaign.log") end=$(date -u +%FT%TZ)" >> "$o/summary.txt"
fi
clk=$(ls -d /sys/class/tenstorrent/tenstorrent\!$card)/tt_aiclk
( while :; do echo "$(date +%s) $(tr -d ' ' < "$clk")"; sleep 1; done ) > "$o/aiclk.txt" &
sampler=$!
t0=$(date +%s)
timeout 3000 $PY -u scripts/release_gate.py --model protenix-v1 --model opendde \
    --journal "$PWD/$o/journal.jsonl" > "$o/gate.log" 2>&1
grc=$?
kill $sampler
echo "gate_rc=$grc gate_wall=$(( $(date +%s) - t0 )) end=$(date -u +%FT%TZ)" >> "$o/summary.txt"
echo "aiclk_mhz min/median/max=$(awk '{print $2}' "$o/aiclk.txt" | sort -n | awk '{a[NR]=$1} END{print a[1]"/"a[int((NR+1)/2)]"/"a[NR]}')" >> "$o/summary.txt"
grep -E "^(protenix-v1|opendde) " "$o/gate.log" >> "$o/summary.txt"
grep -E "\[(protenix-v1|opendde)\].*(seconds|s wall|folded)" "$o/gate.log" | tail -4 >> "$o/summary.txt"
