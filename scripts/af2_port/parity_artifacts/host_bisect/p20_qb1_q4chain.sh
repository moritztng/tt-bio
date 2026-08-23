#!/bin/bash
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p20
PY=/home/ttuser/tt-bio-dev/env/bin/python3
H=$WT/scripts/af2_port/parity_artifacts/host_bisect
cd $WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p20
export OMP_NUM_THREADS=16 PYTHONPATH=$WT
echo "=== Q4 qb1 template-card FORCE_GRID=11,10 $(date -u) ==="
TT_BIO_FORCE_GRID=11,10 $PY -u scripts/af2_port/tap_gate.py --device --triatt-fused none \
  > $H/p20_qb1_tmplcard_grid11x10.json 2> /tmp/p20/q4.err
echo "Q4_EXIT=$?"
touch /tmp/p20/Q4_DONE
for POP in designpop_bg119 designpop_pxd196; do
  A=$WT/scripts/af2_port/parity_artifacts/$POP
  echo "=== designpop none $POP $(date -u) ==="
  $PY -u scripts/af2_port/filter_tolerance.py --mode designs --arm device --stage complex \
    --triatt-fused none --population $A/population.jsonl --pdb-dir $A \
    --out $A/scores_device_p20_none.jsonl > /tmp/p20/ft_$POP.log 2>&1
  echo "FT_${POP}_EXIT=$?"
  touch /tmp/p20/FT_${POP}_DONE
done
echo "=== chain done $(date -u) ==="
touch /tmp/p20/Q4CHAIN_DONE
