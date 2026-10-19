#!/usr/bin/env bash
# BC4 phase 0 rule 2: the fair TT number, PROTOCOL.md exactly. One argument: the run directory.
# Clock probe:  bc4_fair_tt.sh <out> 1      (2 trajectories, verifies AICLK under load after the fw flash)
# Headline:     bc4_fair_tt.sh <out> 9      (10 trajectories, >= 5 warm admissible)
set -u
OUT=${1:?run dir}; WARM=${2:-9}; shift 2 2>/dev/null || shift
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bc4-fair-tt
export JAPANFOLD_BC2_AF2_WEIGHTS="${JAPANFOLD_BC2_AF2_WEIGHTS:-$HOME/bcx_shipped/af2_params}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
uptime > "$OUT/loadavg.start"
~/.local/bin/tt-smi -s > "$OUT/tt-smi.start.json" 2>/dev/null
exec "${BC4_PY:-$HOME/bcx_hostcut_venv/bin/python3}" perf/spd/bench.py --out "$OUT" --chip 0 --model bindcraft2 \
    --bc2 "$HOME/bcx_shipped/bc2" --seed 101 --warm "$WARM" --inputs 80 "$@"
