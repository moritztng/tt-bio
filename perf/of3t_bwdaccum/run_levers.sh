#!/usr/bin/env bash
# One device arm per lever, each a fresh process (one device context per process).
set -u
cd /home/ttuser/.coworker/wt/of3t-bwdaccum
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_bwdaccum
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
for L in "$@"; do
  echo "=== lever $L ==="
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-bwdaccum \
  timeout 600 python3 perf/of3t_bwdaccum/dev_cot.py --lever "$L" \
    --cot-out "$O/devcot_${L}_c64.pt" --ln-capture 0,12,24,36,47 \
    --ln-out "$O/ln_${L}_c64.pt" \
    --boundary "$B" --cap-last "$C" \
    --out "$O/dev_${L}_c64.pt" \
    --report "perf/of3t_bwdaccum/DEV_${L}_c64.json" --arm flipped 2>&1 \
    | grep -E '^\{|OUR GRADIENT|Error|error|Traceback' | tail -6
done
