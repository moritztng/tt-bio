#!/bin/bash
# full_parity_gate.py --legacy-rdx for one arm, the five v0.6.1 legs this gate covers.
#   bash perf/w6_gate/fpg.sh <ARM>
# protenix-hsa-msa is DROPPED from this sweep (585 aa x 5 samples x 5 seeds dominates the wall
# clock). WARROOM/plan rule: drop a whole leg and log it, never cut seeds. Logged here and in
# the state doc, not silently.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-fold-parity-gate || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-fold-parity-gate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
ARM=$1
$PY perf/w6_gate/arm.py --arm "$ARM" >/dev/null || exit 1
$PY scripts/full_parity_gate.py --legacy-rdx --workers tt-quietbox:0 \
    --leg protenix-prot-msa --leg protenix-ubq-msa \
    --leg opendde-trpcage-nomsa --leg opendde-prot-prod --leg opendde-abag \
    --workdir "$HOME/w6_fpg_${ARM}" --out "perf/w6_gate/out/fpg_${ARM}.json" \
    > "perf/w6_gate/out/fpg_${ARM}.log" 2>&1
echo "fpg $ARM rc=$?"
tail -30 "perf/w6_gate/out/fpg_${ARM}.log"
$PY perf/w6_gate/arm.py --arm BASE >/dev/null
