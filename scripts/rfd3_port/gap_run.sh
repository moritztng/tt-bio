#!/bin/bash
# gap_run.sh CHIP OUT TIMEOUT_S CMD...: run CMD (from the tree root) on one chip under its flock, with every TT node's
# AICLK and the load sampled every 0.5 s DURING the run. OUT/run.log (start/end, rc), OUT/harness.log (CMD's output),
# OUT/aiclk.tsv (epoch, node, MHz, load), OUT/nodes.txt (the device node CMD opened: its aiclk.tsv column).
# PY = interpreter with the serving deps (default python, after a Galaxy's env.sh). LOCK = the chip's flock file
# (default ~/spd/locks/chip<CHIP>.lock), WAIT = seconds to queue on it. CMD may say $PY.
set -u
CHIP=$1 OUT=$2 TMO=$3; shift 3
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
mkdir -p "$OUT" ~/spd/locks ~/spd/spd-gap-whrfd3/leases
export PY=${PY:-python} PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-gap-whrfd3 TT_BIO_LEASE_DIR=$HOME/spd/spd-gap-whrfd3/leases
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
flock -w "${WAIT:-600}" 9 || { say "flock busy"; exit 3; }
say "start chip $CHIP tree $(cat TREE_SHA 2>/dev/null || git rev-parse --short HEAD) host $(hostname) cmd: $*"
( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do
    printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done ) 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
timeout -s TERM $((TMO + 120)) timeout -s INT "$TMO" bash -c "$*" > "$OUT/harness.log" 2>&1 &
H=$!
( while kill -0 $H 2>/dev/null; do sleep 5
    for p in $(pstree -p $H | grep -o "([0-9]*)" | tr -d "()"); do ls -l /proc/$p/fd 2>/dev/null | grep -o 'tenstorrent/[0-9]*'; done | sort -u > "$OUT/nodes.tmp"
    [ -s "$OUT/nodes.tmp" ] && { mv "$OUT/nodes.tmp" "$OUT/nodes.txt"; break; }
  done; rm -f "$OUT/nodes.tmp" ) 9>&- &
wait $H; rc=$?
kill $SAMP 2>/dev/null
say "end rc=$rc"
exit $rc
