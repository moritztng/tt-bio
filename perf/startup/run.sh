#!/bin/bash
# Cold-start benchmark of the tt-bio models on one chip: perf/startup/run.sh CHIP OUTROOT [ROUNDS] [MODELS]
# Each run is a fresh process under perf/startup/trace (ST_OUT). Per round and model: a cold run against an empty
# TT_METAL_CACHE, then a warm run against the cache that cold run filled. Before every run the page cache of the
# model's weight files (recorded by earlier runs) is dropped with posix_fadvise, so the weight read is a disk read.
# Rounds go model by model, so a chain cut off early still covers every model. AICLK of every node each second in
# OUTROOT/aiclk.log. Stops when book.py says another owner holds the chip.
set -u
CHIP=$1 OUT=$2 ROUNDS=${3:-3}
MODELS=${4:-"esmc esmfold2 boltz2 protenix-v2 opendde openfold3 rfd4 bindcraft2"}
WT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv-bfcfb20bff6b/bin/python}
BCPY=${BCPY:-$HOME/bcx_e2e_venv/bin/python} BC2=${BC2:-$HOME/bcx_e2e/bc2}
OWNER=${OWNER:-fleet-startup-time-report}
mkdir -p "$OUT"; cd "$WT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
( while :; do echo "$(date +%s.%N | cut -c1-14) $(for f in /sys/class/tenstorrent/tenstorrent!*/tt_aiclk; do n=${f%/tt_aiclk}; echo -n "${n##*!}:$(cut -d' ' -f1 "$f") "; done)"; sleep 1; done ) >> "$OUT/aiclk.log" 2>/dev/null &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT
evict() {  # drop the page cache of every weight file this model read in any earlier run
  cat "$OUT"/$1/*/reads.txt 2>/dev/null | sort -u | "$PY" -c '
import os, sys
for p in sys.stdin.read().split():
    paths = [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs] if os.path.isdir(p) else [p]
    for q in paths:
        try:
            fd = os.open(q, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass'
}
cmd() {  # model, run dir -> argv
  local m=$1 d=$2
  case $m in
    esmc) echo "$PY perf/startup/esmc_run.py $HOME/spd-data/inputs/c730.yaml $d/esmc.jsonl" ;;
    rfd4) echo "$PY perf/bmk/design_cell.py --model rfd4 --repeat 1 --out $d/result.json" ;;
    bindcraft2) echo "$BCPY perf/spd/bench.py --out $d/bench --chip $CHIP --model bindcraft2 --bc2 $BC2 --warm 1 --no-coords" ;;
    *) echo "$PY perf/spd/bench.py --out $d/bench --chip $CHIP --model $m --inputs c730 --warm 1 --no-coords" ;;
  esac
}
say "start chip $CHIP sha $(git rev-parse --short HEAD) dirty $(git status --porcelain -uno | wc -l) host $(hostname) rounds $ROUNDS models [$MODELS]"
for r in $(seq "$ROUNDS"); do
  for m in $MODELS; do
    cache=$OUT/jit/$m-r$r
    for kind in cold warm; do
      d=$OUT/$m/r$r-$kind
      [ -f "$d/rc" ] && continue
      holder=$("$PY" perf/spd/book.py active "$CHIP" --except "$OWNER" 2>/dev/null)
      if [ -n "$holder" ]; then say "chip $CHIP held by $holder, stopping"; exit 4; fi
      rm -rf "$d"; mkdir -p "$d"
      [ $kind = cold ] && rm -rf "$cache"
      mkdir -p "$cache"
      evict "$m"
      extra=; [ $m = bindcraft2 ] && extra=:$BC2
      say "$m r$r $kind begin"
      t0=$(date +%s.%N)
      ( cd "$WT"; env ST_OUT="$d/trace.jsonl" PYTHONPATH="$WT/perf/startup/trace:$WT$extra" TT_METAL_CACHE="$cache" \
          TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP TT_BIO_LEASE_HOLDER=worker:startup-time-report \
          timeout -s INT 2400 $(cmd $m $d) > "$d/log" 2>&1 ) &
      pid=$!
      ( sleep 90; for p in $(pgrep -f "$d/"); do ls -l /proc/$p/fd 2>/dev/null; done | grep -o "tenstorrent/[0-9]*" | sort -u > "$d/nodes.txt" ) &
      wait $pid; rc=$?
      t1=$(date +%s.%N)
      echo "$t0 $t1" > "$d/wall"; echo $rc > "$d/rc"
      "$PY" -c 'import json,sys
for l in open(sys.argv[1]):
    e=json.loads(l)
    if e.get("ev")=="summary": print("\n".join(e["reads"]))' "$d/trace.jsonl" > "$d/reads.txt" 2>/dev/null
      say "$m r$r $kind rc=$rc wall=$(echo "$t1 - $t0" | bc) cache=$(du -sh "$cache" | cut -f1)"
    done
  done
done
rm -rf "$OUT/jit"
say "end"
date -u +%FT%TZ > "$OUT/done"
