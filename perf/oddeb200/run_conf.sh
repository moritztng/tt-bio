#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
echo "### screen_confidence start $(date -Is)"
/home/ttuser/tt-bio-dev/env/bin/python3 -u perf/oddeb200/screen_confidence.py
echo "RC=$?"
echo "### done $(date -Is)"
