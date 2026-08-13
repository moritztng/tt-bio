#!/bin/sh
# Re-anchor RFD3 on the pinned representative fixture rfd3_R4 (6051 atoms, 685 tokens).
# Three ledger runs in one benchlock hold, shipped defaults (RFD3_SPARSE_BIAS and
# RFD3_TUNE_MATMUL are both opt-in and stay off here -- this is what a user gets today).
set -e
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PDB=perf/dsfix/targets/R4_9q6y_A.pdb
CONTIG=A1-585,100
N=10
mkdir -p perf/p42

run() {
  tag=$1; shift
  echo "=== $tag ==="
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture \
      PYTHONPATH=/home/ttuser/.coworker/wt/rfd3-optimize-on-fixture \
      $PY scripts/rfd3_port/p35_host_ledger.py \
      --pdb $PDB --contig $CONTIG --num_timesteps $N --seed 7 \
      --out perf/p42/$tag.json "$@" 2>&1 \
    | grep -v -E "info \||Fabric|topology|Degree|Config\{|DEBUG|loguru|Always | ID \|"
}

run r4_b2_plain  --designs 2 --plain
run r4_b2_ledger --designs 2
run r4_b1_plain  --designs 1 --plain
echo "=== ALL DONE ==="
