#!/bin/bash
# lpx-matmul on .107 (UF-EV-A4-GWH01), on the chip lpx-orchestrator carved for this row (state/lpx/CHIPS.md).
# The orchestrator's keeper holds the agent-side lease, so this never touches the agent: the bench opens
# the device under a PRIVATE lease dir. One process on the chip at a time.
cd ~/lpx-matmul; . ~/japanfold/env.sh
RUN=${1:?run dir}; CHIP=${2:?chip}; BUDGET=${3:-3600}; TOP=${4:-16}
export PATH="$PATH:$HOME/japanfold/msa/venv/bin" TT_BIO_LEASE_HOLDER=lpx-matmul \
       TT_BIO_LEASE_DIR=$HOME/lpx/lpx-matmul/leases TT_BIO_LEASE_TIMEOUT=600
mkdir -p "$TT_BIO_LEASE_DIR" "$RUN"; ln -sfn ~/pfm-ttfast/msa "$RUN/msa"
say(){ echo "$(date -u +%FT%TZ) $*"; }
grep -q "held by lpx-matmul" ~/lpx/log/keeper-$CHIP.log || { say "chip $CHIP is not carved for lpx-matmul"; exit 3; }
pgrep -f "bench_wh.py .*lpx-matmul" >/dev/null && { say "a lpx-matmul bench is already running"; exit 4; }
say "start engine $(git -C ~/japanfold/engine rev-parse HEAD) load $(cat /proc/loadavg)"
timeout 14400 python bench_wh.py "$RUN" "$CHIP" 31 complex730.yaml "$BUDGET" "$TOP"
say "rc=$? load $(cat /proc/loadavg)"
