#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
echo "### conf_phases start $(date -Is)"
/home/ttuser/tt-bio-dev/env/bin/python3 -u perf/oddeb200/screen_conf_phases.py
echo "RC=$?"
echo "### done $(date -Is)"
