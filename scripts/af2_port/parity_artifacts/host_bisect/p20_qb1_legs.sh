#!/bin/bash
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p20
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p20
export OMP_NUM_THREADS=16 PYTHONPATH=$WT
echo "=== Q1 template-host $(date -u) ==="
nice -n 5 $PY -u scripts/af2_port/tap_gate.py --device --triatt-fused none --template-host \
  > /tmp/p20/q1_template_host.json 2> /tmp/p20/q1_template_host.err
echo "Q1_EXIT=$?"
echo "=== Q2 template-card $(date -u) ==="
nice -n 5 $PY -u scripts/af2_port/tap_gate.py --device --triatt-fused none \
  > /tmp/p20/q2_template_card.json 2> /tmp/p20/q2_template_card.err
echo "Q2_EXIT=$?"
echo "=== done $(date -u) ==="
