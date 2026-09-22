#!/usr/bin/env bash
# One trunk-scope gradient arm on the tree that SHIPS the repair: wk/of3t, which carries
# d116's unified softmax_bw_inner and d56-renorm's default-on flip.
#
#   CTRL  RENORM=0        the pre-repair rule, i.e. what origin/main still computes
#   SHIP  RENORM=default  no env var set at all: what a training run actually gets
#   ON    RENORM=1        the flag forced, to separate "default" from "on"
#   BREAK RENORM=default --permute-cot SEED
#
# --arm flipped because that is the arm whose forward passes A18 and the arm the
# 9.025172e+00 of record was taken on (of3t-trunkg043, of3t-apbgrad).
set -u
WT=/home/ttuser/.coworker/wt/of3t-d116-verify
cd "$WT"
source /home/ttuser/tt-bio-dev/env/bin/activate
O=/home/ttuser/of3t_d116_verify
B=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
TAG=$1; RENORM=$2; shift 2
if [ "$RENORM" = "default" ]; then unset TT_BIO_SOFTMAX_BW_RENORM; else export TT_BIO_SOFTMAX_BW_RENORM=$RENORM; fi
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-d116-verify
export TT_BIO_RENORM_STATS_DIR="$O/stats_${TAG}"
rm -rf "$TT_BIO_RENORM_STATS_DIR"; mkdir -p "$TT_BIO_RENORM_STATS_DIR"
timeout 3000 python3 perf/of3t_trunkg043/dev_grad.py \
  --boundary "$B" --cap-last "$C" --arm flipped \
  --out "$O/dev_${TAG}_c64.pt" \
  --report "perf/of3t_d116_verify/DEV_${TAG}_c64.json" "$@" > "$O/${TAG}.log" 2>&1
rc=$?
grep -E '^\{|OUR GRADIENT|Traceback|rror' "$O/${TAG}.log" | tail -8
echo "rc=$rc"
cat "$TT_BIO_RENORM_STATS_DIR"/*.json 2>/dev/null
exit $rc
