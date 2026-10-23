#!/bin/bash
# One RFD4 leg on one Wormhole chip of a Galaxy, under the chip flock, with a private lease and the
# AICLK of every node sampled during the run. Mirrors ~/spd/spd-orchestrator/run_cell.sh, which every
# other SPD row on these boxes uses; a second sampler would make the numbers incomparable.
#   CHIP=<umd> bash perf/rfd4/wh/run_leg.sh <outdir> <timeout_s> -- <cmd...>
set -u
OUT=$1 TMO=$2; shift 3
T=$(cd "$(dirname "$0")/../../.." && pwd)
R=$HOME/spd/rfd4-wh
mkdir -p "$OUT" $R/leases ~/spd/locks
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP
export TT_BIO_LEASE_HOLDER=rfd4-wh-bringup TT_BIO_LEASE_DIR=$R/leases
eval "$(cd $T && python -c "from tt_bio import runtime as r; [print(f'export {k}={v}') for k,v in r.host_thread_cap_env(${SHARE:-32}, None).items()]")"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> ~/spd/locks/chip$CHIP.lock
flock -w "${WAIT:-600}" 9 || { say "flock busy"; exit 3; }
say "start chip $CHIP tree $(cat $T/SHA 2>/dev/null) host $(hostname) cmd: $*"
( while kill -0 $$ 2>/dev/null; do l=$(cut -d' ' -f1 /proc/loadavg)
    for n in /sys/class/tenstorrent/tenstorrent\!*; do
      printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done
    sleep 0.5; done ) 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
cd $T
timeout -s TERM $((TMO + 120)) timeout -s INT "$TMO" "$@" > "$OUT/harness.log" 2>&1 &
H=$!
wait $H; rc=$?
kill $SAMP 2>/dev/null
say "end rc=$rc"
exit $rc
