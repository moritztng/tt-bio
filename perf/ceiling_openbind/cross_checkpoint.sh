# Walk the OTHER checkpoint on the same trunk, on the same card, once the OpenBind verification
# chain is done.
#
#   sh perf/ceiling_openbind/cross_checkpoint.sh <tree> <out-dir> <wait-on-log>
#
# OpenBind-0 and OpenFold3 are one OF3Trunk with two sets of weights. A capacity fix in that trunk
# is a SHARED fix or it is a per-model patch, and the only thing that tells them apart is walking
# the other checkpoint on the same engine. ws:ceiling-openfold3-1024 reports OpenFold3 reaching 896
# and failing 960/1024 on a 1.86 GB request -- 14191 x 1024 x 64 bf16, the MSA block's third
# residual copy, which this branch removed. So this leg is a prediction with a number attached, not
# a survey.
set -u
TREE=$1; OUT=$2; WAIT=$3
waited=0
while ! grep -q "^VERIFY DONE" "$WAIT" 2>/dev/null; do
  waited=$((waited + 1))
  if [ "$waited" -gt 360 ]; then
    echo "GIVING UP waiting on $WAIT $(date -u +%FT%TZ)" >> "$OUT/ladder.log"
    exit 2
  fi
  sleep 30
done
mkdir -p "$OUT"
MODEL=openfold3 \
RUNGS=/home/cust-team/mthuening/ceilof3/rundir/msafix_tile \
MSA=/home/cust-team/mthuening/ceilof3/rundir/msacache_deep \
  sh "$TREE/perf/ceiling_openbind/run_ladder.sh" "$TREE" "$OUT" tile_1024 tile_960 tile_896
