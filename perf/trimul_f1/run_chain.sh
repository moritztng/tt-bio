#!/usr/bin/env bash
# F1 release-gate chain: parity sweep, per-model censuses, openfold3 control, esmfold2 timing.
set -u
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOG=perf/trimul_f1/logs
run() { echo "=== LEG $1 start $(date -u +%H:%M:%S) ==="; shift; "$@"; echo "=== LEG exit=$? $(date -u +%H:%M:%S) ==="; }
run parity  $PY perf/trimul_f1/f1_parity.py                                > $LOG/chain_parity.log 2>&1
run parityL $PY perf/trimul_f1/f1_parity.py 640 768 1024 >> $LOG/chain_parity.log 2>&1
run boltz2  $PY perf/size512/fold_ab512.py --model boltz2  --sizes 512 --arms nof1,f1 --out perf/trimul_f1/census_boltz2_512_qb2c2.json  > $LOG/chain_boltz2.log 2>&1
run opendde $PY perf/size512/fold_ab512.py --model opendde  --sizes 512 --arms nof1,f1 --out perf/trimul_f1/census_opendde_512_qb2c2.json  > $LOG/chain_opendde.log 2>&1
run esmfold $PY perf/size512/fold_ab512.py --model esmfold2 --sizes 512 --arms nof1,f1,nof1,f1,nof1,f1 --out perf/trimul_f1/timing_esmfold2_512_qb2c2.json > $LOG/chain_esmfold2.log 2>&1
run of3ctrl $PY perf/other512/fold_ab_multi.py --model openfold3 --sizes 512 --arms on --out perf/trimul_f1/control_openfold3_512_qb2c2.json > $LOG/chain_of3ctrl.log 2>&1
echo "=== CHAIN DONE $(date -u +%H:%M:%S) ==="
