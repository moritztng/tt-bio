#!/bin/sh
# Round 2. Closes the two gaps the first pass named: the op site walls were measured once, and the
# release-gated in0_block_w=8 arm had never been run in a fold. Four runs in ONE session, with the
# unmodified baseline arm run first AND last so process-to-process drift is bracketed rather than
# assumed away.
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-l1-output
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output
for spec in "base2 0 0 1" "both2 1 1 1" "rg 1 1 16" "base3 0 0 1"; do
  # shellcheck disable=SC2086
  set -- $spec
  echo "########## OPS $1  l1_out=$2 bias=$3 l1_bw=$4  $(uptime | sed 's/.*load average/load average/') ##########"
  python3 perf/p3l1/fold_ab.py --arm "$1" --l1-out "$2" --bias-l1-norm "$3" --l1-bw "$4" \
    --instrument ops --repeat 1 --out "perf/p3l1/ops_$1.json" 2>&1 \
    | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning" \
    | grep -vE "^ +[0-9]+,?$" | tail -28
done
echo "########## ROUND 2 DONE  $(uptime | sed 's/.*load average/load average/') ##########"
