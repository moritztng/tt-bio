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
# The base the arms are built from, not HEAD -- HEAD moves with every commit on this branch
# and would throw away a resumable workdir each time.
BASE_REF=$(sed -n 's/^BASE_REF = "\(.*\)"/\1/p' perf/w6_c2fix/arm.py)
[ -n "$BASE_REF" ] || exit 1
$PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
# Three things this command line has to get right, each of which has already cost a run:
#  - locality is decided against socket.gethostname(), which is tt-quietbox here, not qb1,
#    so --workers qb1:N is dialled as a remote and every leg dies with `predict exited 255`;
#  - NO --seeds. It takes a comma-separated LIST, not a count. `--seeds 5` parses as [5], no
#    fixture carries a seed 5, and every leg comes back BLOCKED-REF-REGEN-NEEDED under a
#    GATE PASS banner. Omitting it uses each leg's own recorded seeds, 0..4 for the structure
#    legs and (0,) for opendde-abag, so a literal 0,1,2,3,4 would be wrong for that leg too;
#  - the workdir is stamped with the base ref. full_parity_gate reuses a leg report it finds
#    there, so an unstamped dir lets a leg scored on the previous main tip survive a rebase
#    and be reported as if it had been measured on the new one. Resume still works within a
#    base. Keep every comment out of the command itself: a `#` line inside a backslash
#    continuation swallows the rest of the logical line, and `bash -n` does not flag it.
$PY scripts/full_parity_gate.py --legacy-rdx --workers tt-quietbox:1 \
    --leg protenix-prot-msa --leg protenix-ubq-msa \
    --leg opendde-prot-prod --leg opendde-abag --leg boltz2-hsa-nomsa \
    --workdir "$HOME/c2fix_fpg_${ARM}_${BASE_REF}" \
    --out perf/w6_c2fix/out/fpg_${ARM}.json 2>&1 | tee perf/w6_c2fix/out/fpg_${ARM}.log
echo "FPG DONE $ARM $(date -u +%H:%M:%S)"
