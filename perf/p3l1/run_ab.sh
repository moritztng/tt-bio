#!/bin/sh
# p3-l1-output — the A/B. Four arms, each in its own process on the same card in the same
# session, so deliverable 1 (the L1 output) and deliverable 2 (the L1 layer_norm source) carry
# separate numbers instead of one bundled one. `$1 $2 $3` = label, _PAIR_PROJ_L1_OUT,
# _PAIR_BIAS_L1_NORM. Host load is printed around every arm.
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-l1-output
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output
R="${1:-1}"
for spec in "base 0 0" "l1out 1 0" "bias 0 1" "both 1 1"; do
  # shellcheck disable=SC2086
  set -- $spec
  echo "########## ARM $1 round $R  $(uptime | sed 's/.*load average/load average/') ##########"
  python3 perf/p3l1/fold_ab.py --arm "$1_r$R" --l1-out "$2" --bias-l1-norm "$3" --l1-bw 1 \
    --instrument block --repeat 3 --out "perf/p3l1/ab_$1_r$R.json" 2>&1 \
    | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning" \
    | tail -26
done
echo "########## A/B ROUND $R DONE  $(uptime | sed 's/.*load average/load average/') ##########"
