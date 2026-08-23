#!/bin/bash
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p20-qb2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p20
export OMP_NUM_THREADS=16 PYTHONPATH=$WT
echo "=== Q3 qb2 template-card $(date -u) ==="
$PY -u scripts/af2_port/tap_gate.py --device --triatt-fused none \
  > /tmp/p20/q3_template_card.json 2> /tmp/p20/q3.err
echo "Q3_EXIT=$?"
echo "=== Q3ctl qb2 template-host (reproduces p19_committed) $(date -u) ==="
$PY -u scripts/af2_port/tap_gate.py --device --triatt-fused none --template-host \
  > /tmp/p20/q3ctl_template_host.json 2> /tmp/p20/q3ctl.err
echo "Q3CTL_EXIT=$?"
echo "=== done $(date -u) ==="
touch /tmp/p20/Q3_ALL_DONE
