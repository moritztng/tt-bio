#!/usr/bin/env bash
# One scope arm per invocation, fresh process, one device context.
# Mirrors perf/of3t_apbgrad/run_scope.sh exactly except for the lever, the output paths and
# the worktree: that script cd's to /home/ttuser/.coworker/wt/of3t-apbgrad, which fleet
# hygiene has already removed.
set -u
WT=/home/ttuser/.coworker/wt/of3t-lnaffine
cd "$WT" || exit 1
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/tmp/of3t/lnaffine
mkdir -p "$O" "$WT/perf/of3t_lnaffine"
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
TAG=$1; LEVER=$2; shift 2
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-lnaffine \
timeout 2400 python3 perf/of3t_bwdaccum/dev_cot.py --lever "$LEVER" \
  --ln-capture 0,12,24,36,47 --ln-out "$O/ln_${TAG}_c64.pt" \
  --boundary "$B" --cap-last "$C" --out "$O/dev_${TAG}_c64.pt" \
  --report "$WT/perf/of3t_lnaffine/DEV_${TAG}_c64.json" --arm flipped "$@" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|Traceback|Error|error' | tail -8
