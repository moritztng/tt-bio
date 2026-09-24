#!/bin/bash
# card 2, this row's lease identity; prints only the harness lines
cd /home/ttuser/.coworker/wt/bcx-nan || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-nan
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
/home/ttuser/bcx_e2e_venv/bin/python perf/bcx_nan/carried.py "$@" 2>&1 | grep -E "^(BLOCKS|POISON|START|REP|SUMMARY|.*TT_FATAL|Traceback|  File|[A-Za-z]*Error)"
