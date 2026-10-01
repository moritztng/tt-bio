#!/bin/bash
# One arm of the acceptance pair: does BindCraft 2 still find binders at a large token axis?
#
# Both arms are the SAME target (EGFR), the same epitope, the same 150 aa binder, the same seed,
# the same tree and the same board. The only thing that differs is how much of the receptor is in
# the box: arm A is the domain III crop a customer makes today, arm B is the whole ectodomain.
# Comparing EGFR at 768 against the PD-L1 rate at 288 would confound the target with the axis.
#
# No TT_BIO_* lever is set. This is the shipping path, query loop off, and the wall-clock is the
# price of that choice: a kernel that changes the gradient above 320 tokens would make a zero
# ambiguous between the size and the kernel.
#
#   campaign.sh <arm: a|b> <card> <tag> [seed]
set -uo pipefail
arm=$1; card=$2; tag=$3; seed=${4:-100}
cd /home/ttuser/.coworker/wt/bcw-accept
case "$arm" in
  a) target=hEGFR_d3; limit=43200 ;;
  b) target=hEGFR;    limit=129600 ;;
  *) echo "arm must be a or b"; exit 2 ;;
esac
o=perf/bcw_accept/out; mkdir -p "$o/$tag"
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$o/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-accept
# Co-tenants, because qb2 carries bcw-land and bcp-evo and a round time taken beside them has to
# record what else was on the box.
( while :; do { echo "--- $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)";
    ps -eo pid,etime,pcpu,args --sort=-pcpu | head -5 | cut -c1-150; } >> "$o/$tag.cotenants";
    sleep 180; done ) &
trap "kill $! 2>/dev/null" EXIT
echo "=== $(date -u +%FT%TZ) arm=$arm target=$target card=$card seed=$seed head=$(git rev-parse --short HEAD)" >> "$o/campaign.log"
timeout $limit /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
    --target "$target" --binder 150 --rounds 0 \
    --trajectories auto --max-trajectories 10 --final-designs 10 --seed "$seed" \
    --params /home/ttuser/bcx_e2e/af2_params --out "$o/$tag" > "$o/$tag.log" 2>&1
echo "=== rc=$? $(date -u +%FT%TZ) arm=$arm $tag" >> "$o/campaign.log"
