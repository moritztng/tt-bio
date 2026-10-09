#!/bin/bash
# spd-serve on .107 chip CHIP (CHIPS.md): arms one after another, SIGINT then SIGTERM. The chip flock is held for
# the whole run, not per arm: between two arms the box's pool (~/spd/pool) reads a free flock as an idle chip.
set -u
CHIP=$1; ROOT=$2; shift 2
mkdir -p "$ROOT"
. ~/japanfold/env.sh > /dev/null 2>&1
S=~/spd/spd-serve
BENCH=${BENCH:-$S/bench}            # a later run gets its own copy, so a running one keeps its bench
export TT_BIO_LEASE_HOLDER=spd-serve TT_BIO_LEASE_DIR=$S/leases TT_METAL_CACHE=$S/cache-$CHIP
mkdir -p "$TT_BIO_LEASE_DIR" "$TT_METAL_CACHE" ~/spd/locks
say(){ echo "$(date -u +%FT%TZ) $*" >> "$ROOT/run.log"; }
exec 9> ~/spd/locks/chip$CHIP.lock
flock -w "${WAIT:-7200}" 9 || { say "chip $CHIP flock not free after ${WAIT:-7200} s"; exit 1; }
# each arg: NAME|ENGINE(- = installed prod engine)|EXTRA ARGS
for spec in "$@"; do
  IFS='|' read -r name eng extra <<< "$spec"
  E=(); [ "$eng" != "-" ] && E=(--engine "$S/$eng")
  say "arm $name engine $eng extra '$extra' start"
  timeout -s TERM 7260 timeout -s INT 7200 \
    python $BENCH/serve_bench.py --out "$ROOT/$name" --chip $CHIP "${E[@]}" $extra > "$ROOT/$name.log" 2>&1
  say "arm $name rc=$?"
done
say "all done"
