#!/bin/bash
# lpx-e2e on .107: one bench process on the chip lpx-orchestrator holds for this row (state/lpx/CHIPS.md).
# The keeper holds tt-bio's flock in the agent's lease dir, so this process takes a PRIVATE lease dir and
# never stops the agent. ENGINE is a tt_bio tree from wk/lpx-e2e, put ahead of the installed engine.
# usage: run.sh RUN CHIP PLAN
cd ~/lpx-e2e; . ~/japanfold/env.sh
RUN=${1:?run dir}; CHIP=${2:?chip}; PLAN=${3:?plan}
export PATH="$PATH:$HOME/japanfold/msa/venv/bin" PYTHONPATH=~/lpx-e2e/engine \
       TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_DIR=~/lpx/lpx-e2e/leases TT_BIO_LEASE_HOLDER=lpx-e2e
mkdir -p "$RUN" "$TT_BIO_LEASE_DIR"; ln -sfn ~/lpx-e2e/msa "$RUN/msa"
say(){ echo "$(date -u +%FT%TZ) $*"; }
say "start engine $(cat engine/HEAD) chip $CHIP plan $PLAN load $(cat /proc/loadavg)"
timeout 14400 python bench_wh.py "$RUN" "$CHIP" 31 complex730.yaml "$PLAN" > "$RUN.log" 2>&1
say "rc=$? end load $(cat /proc/loadavg)"
