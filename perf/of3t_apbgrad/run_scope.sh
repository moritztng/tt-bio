#!/usr/bin/env bash
# Scope arm: no teacher forcing, so the number is comparable to of3t-bwdaccum's 9.025172e+00.
set -u
cd /home/ttuser/.coworker/wt/of3t-apbgrad
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_apbgrad
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
TAG=$1; RENORM=$2; shift 2
TT_BIO_SOFTMAX_BW_RENORM=$RENORM \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-apbgrad \
timeout 1800 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  --cot-out "$O/devcot_scope_${TAG}_c64.pt" --ln-capture 0,12,24,36,47 \
  --ln-out "$O/ln_scope_${TAG}_c64.pt" \
  --boundary "$B" --cap-last "$C" --out "$O/dev_scope_${TAG}_c64.pt" \
  --report "perf/of3t_apbgrad/DEV_SCOPE_${TAG}_c64.json" --arm flipped "$@" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|Traceback|Error|error' | tail -5
