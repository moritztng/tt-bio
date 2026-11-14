#!/bin/bash
# spd-bh: the lpx-e2e fold bench (perf/lpx_e2e/bench_wh.py, arch-neutral despite its name) on ONE
# Blackhole chip of qb1 (p150a) or qb2 (p300c), engine = this worktree. 730-token fixture, MSA cache at
# depth 9,947 copied from .107 (~/spd-bh/msa), 5 samples, 10 recycles, 200 steps, AICLK sampled
# during every fold. Takes the chip only if nothing holds its device node.
# usage: run.sh RUN CHIP PLAN      e.g. run.sh ~/spd-bh/runs/p150a-base 2 "exact:101,102,103;lpx:101,102,103"
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?run dir}; CHIP=${2:?logical chip}; PLAN=${3:?plan}
PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
mkdir -p ~/spd-bh/msa-db-empty
export LPX_MSA_DB=~/spd-bh/msa-db-empty PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-bh
mkdir -p "$RUN"; ln -sfn ~/spd-bh/msa "$RUN/msa"
say(){ echo "$(date -u +%FT%TZ) $*"; }
say "start engine $(git -C "$WT" rev-parse --short HEAD) host $(hostname) chip $CHIP plan $PLAN load $(cat /proc/loadavg)"
cd "$WT/perf/lpx_e2e"
nice -n 5 timeout 21600 "$PY" bench_wh.py "$RUN" "$CHIP" 0 complex730.yaml "$PLAN" > "$RUN.log" 2>&1
say "rc=$? end load $(cat /proc/loadavg)"
