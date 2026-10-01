#!/bin/bash
# bcw-land round sitting: base = 8c3783b87 (main + bcp-evo, the tree main becomes when bcp-evo lands)
# in .base/, stack = this tree (C1+C4+C9 armed by default, slowmode at memory="fast"). PD-L1 binder
# 146, 288 tokens, N by duotraj.auto_trajectories, ABBAAB at the process boundary, one card.
#   round_sit.sh <card> [rounds] [out subdir]
set -uo pipefail
cd "$(dirname "$0")/../.."
here=$PWD; card=$1; r=${2:-9}
o=$here/perf/bcw_land/out/${3:-round}; mkdir -p "$o"
( while :; do echo "--- $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"; ps -eo pid,etime,pcpu,args --sort=-pcpu | head -8 | cut -c1-180; sleep 60; done ) >> "$o/cotenants.txt" &
snap=$!; trap "kill $snap 2>/dev/null" EXIT
py=/home/ttuser/bcx_e2e_venv/bin/python3
export BCX_BC2=/home/ttuser/bcx_e2e/bc2 JAX_COMPILATION_CACHE_DIR=$o/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land
for tag in base1 stack1 stack2 base2 base3 stack3; do
    if [ "${tag%?}" = base ]; then tree=$here/.base; duo=perf/bcx_p10_duotraj/duo_round.py
    else tree=$here; duo=perf/bcw_land/duo_fast.py; fi
    out=$o/$tag; rm -rf "$out"; mkdir -p "$out"
    n=$(cd "$tree" && PYTHONPATH=$tree $py -c "from tt_bio import duotraj; print(duotraj.auto_trajectories(288)[0])")
    echo "=== $tag tree $(git -C "$tree" rev-parse --short HEAD) N=$n load $(cut -d" " -f1 /proc/loadavg) $(date -u +%FT%TZ)"
    (cd "$tree" && PYTHONPATH=$tree $py -u $duo --rounds "$r" --interleave $([ "$n" -gt 1 ] && echo 1 || echo 0) \
        --trajectories "$n" --binder 146 --params /home/ttuser/bcx_e2e/af2_params --out "$out") > "$o/$tag.log" 2>&1 \
        || echo "  $tag exited $?"
done
echo "=== sitting done $(date -u +%FT%TZ)"
$py perf/bcp_device/split.py "$o"/base1 "$o"/stack1 "$o"/stack2 "$o"/base2 "$o"/base3 "$o"/stack3 > "$o/split.txt" 2>&1
