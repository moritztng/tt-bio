#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/of3t-bwdaccum
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_bwdaccum
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
R=$O/refcot_f64_c64.pt
for L in "$@"; do
  echo "=== inject lever $L ==="
  EXTRA=""
  if [ "$L" = "none" ]; then EXTRA="--ln-capture 47,44,42,39,22,15,4,37,36,31,30,16 --ln-out $O/ln_inject_c64.pt"; fi
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-bwdaccum \
  timeout 600 python3 perf/of3t_bwdaccum/dev_cot.py --lever "$L" --inject-ref "$R" \
    --cot-out "$O/devcot_inj_${L}_c64.pt" $EXTRA \
    --boundary "$B" --cap-last "$C" --out "$O/dev_inj_${L}_c64.pt" \
    --report "perf/of3t_bwdaccum/DEV_INJ_${L}_c64.json" --arm flipped 2>&1 \
    | grep -E '^\{|Traceback|Error|error' | tail -5
done
