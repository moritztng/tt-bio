#!/bin/bash
# spd-attn jobs on one chip under its SPD flock, run from this checkout. usage:
#   queue.sh OUT CHIP LOCK PY op            op bench (perf/spd_attn/opbench.py all), ~10 min
#   queue.sh OUT CHIP LOCK PY atom          op bench, atom attention arms only (opbench.py atom), ~3 min
#   queue.sh OUT CHIP LOCK PY steps         atom attention step timings, fp32 then bf16 (atom_steps.py), ~5 min
#   queue.sh OUT CHIP LOCK PY fold          c730 fold, arms off (both levers off) and attn (branch default), 1 cold + 3 warm
# ARMS="attn off:..." replaces the fold arms. DATA=<dir> points bench.py at another spd-data tree; SHARE=32 is the Galaxy host-thread share.
set -u
OUT=$1 CHIP=$2 LOCK=$3 PY=$4 JOB=$5
cd "$(dirname "$0")/../.."
mkdir -p "$OUT" "$HOME/spd/spd-attn/leases"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-attn TT_BIO_LEASE_DIR=$HOME/spd/spd-attn/leases
say(){ echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
say "waiting for $LOCK; head $(git rev-parse --short HEAD)"
exec 9>"$LOCK"
flock -w 10800 9 || { say "flock timeout"; exit 3; }
say "flock held, load $(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$JOB" = op ] || [ "$JOB" = atom ] || [ "$JOB" = roof ]; then
    WHICH=all; [ "$JOB" = atom ] || [ "$JOB" = roof ] && WHICH=$JOB
    timeout -s INT 2400 $PY perf/spd_attn/opbench.py "$OUT/op" "$CHIP" $WHICH > "$OUT/op.log" 2>&1
    say "op rc=$?"
elif [ "$JOB" = steps ]; then
    for dt in fp32 bf16; do
        timeout -s INT 1200 $PY perf/spd_attn/atom_steps.py "$OUT/steps_$dt" "$CHIP" $dt > "$OUT/steps_$dt.log" 2>&1
        say "steps $dt rc=$?"
    done
else
    OFF=TT_BIO_SDPA_FUSED_PADDED=0,TT_BIO_ATOM_SUPERSET_WINDOW=0
    for arm in ${ARMS:-"off:$OFF" attn "off2:$OFF"}; do
        name=${arm%%:*}
        timeout -s INT 5400 $PY perf/spd/bench.py --out "$OUT/$name" --chip "$CHIP" --arm "$arm" --inputs c730 --warm 3 ${DATA:+--data "$DATA"} ${SHARE:+--share "$SHARE"} \
            > "$OUT/$name.log" 2>&1
        say "$name rc=$?"
    done
fi
say "done"
