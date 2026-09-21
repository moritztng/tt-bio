#!/usr/bin/env bash
# One device arm per invocation, fresh process. $1=tag, $2=0|1 for the renorm lever.
set -u
cd /home/ttuser/.coworker/wt/of3t-apbgrad
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_apbgrad
mkdir -p "$O"
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
R=/home/ttuser/of3t_bwdaccum/refcot_f64_c64.pt
TAG=$1; RENORM=$2; shift 2
TT_BIO_SOFTMAX_BW_RENORM=$RENORM \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-apbgrad \
timeout 1800 python3 perf/of3t_bwdaccum/dev_cot.py --lever none --inject-ref "$R" \
  --cot-out "$O/devcot_${TAG}_c64.pt" \
  --ln-capture 47,44,42,39,22,15,4,37,36,31,30,16 --ln-out "$O/ln_${TAG}_c64.pt" \
  --boundary "$B" --cap-last "$C" --out "$O/dev_${TAG}_c64.pt" \
  --report "perf/of3t_apbgrad/DEV_${TAG}_c64.json" --arm flipped "$@" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|Traceback|Error|error' | tail -6
