#!/bin/bash
# full_parity_gate.py --legacy-rdx for one arm, per-leg recorded seeds, the five v0.6.1 legs W11 covered.
# protenix-hsa-msa stays dropped (585 aa x 5 samples x 5 seeds dominates the wall clock) --
# a whole leg dropped and logged, never seeds cut.
#   bash perf/w6_c2fix/fpg.sh <BASE|C2FIX>
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
ARM=$1
$PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
# locality is decided against socket.gethostname(), which is tt-quietbox here, not qb1
# NO --seeds. It takes a comma-separated LIST, not a count: `--seeds 5` parses as [5], no
# fixture has a seed 5, and every leg comes back BLOCKED-REF-REGEN-NEEDED. Omitting it uses
# each leg's own recorded seeds -- 0..4 for the structure legs, (0,) for opendde-abag -- so a
# literal 0,1,2,3,4 would be wrong for that leg too.
$PY scripts/full_parity_gate.py --legacy-rdx --workers tt-quietbox:1 \
    --leg protenix-prot-msa --leg protenix-ubq-msa \
    --leg opendde-prot-prod --leg opendde-abag --leg boltz2-hsa-nomsa \
    --workdir "$HOME/c2fix_fpg_${ARM}" \
    --out perf/w6_c2fix/out/fpg_${ARM}.json 2>&1 | tee perf/w6_c2fix/out/fpg_${ARM}.log
echo "FPG DONE $ARM $(date -u +%H:%M:%S)"
