#!/usr/bin/env bash
# One device arm per invocation, a fresh process each (one device context per process).
set -u
cd /home/ttuser/.coworker/wt/of3t-apbgrad
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_apbgrad
mkdir -p "$O"
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
R=/home/ttuser/of3t_bwdaccum/refcot_f64_c64.pt
TAG=${1:-tap}
shift || true
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-apbgrad \
timeout 1800 python3 perf/of3t_apbgrad/apb_tap.py \
  --tap-out "$O/${TAG}_c64.pt" \
  --lever none --inject-ref "$R" \
  --cot-out "$O/devcot_${TAG}_c64.pt" \
  --boundary "$B" --cap-last "$C" --out "$O/dev_${TAG}_c64.pt" \
  --report "perf/of3t_apbgrad/DEV_${TAG^^}_c64.json" --arm flipped "$@" 2>&1 \
  | grep -vE "^\[|Always|^ *$" | tail -25
