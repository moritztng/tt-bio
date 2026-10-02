#!/bin/bash
# usage: probe.sh '<json cases>' <out.json>
cd /home/ttuser/.coworker/wt/bcw-bmm
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcw-bmm \
  timeout 600 /home/ttuser/bcx_e2e_venv/bin/python3 perf/bcw_bmm/cb_probe.py "$1" "$2" 2>&1 \
  | grep -v "DEBUG\|Config{\| info \|critical"
