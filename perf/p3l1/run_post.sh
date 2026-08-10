#!/bin/sh
# parity at the fold's own shapes, then the ops-attribution pass for base and both.
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-l1-output
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-l1-output
echo "########## GUARD  $(uptime | sed 's/.*load average/load average/') ##########"
python3 perf/p3l1/guard_check.py 2>&1 | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
echo "########## PARITY  $(uptime | sed 's/.*load average/load average/') ##########"
python3 perf/p3l1/parity_c1.py perf/p3l1/parity_c1.json 2>&1 \
  | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
for spec in "base 0 0" "both 1 1"; do
  # shellcheck disable=SC2086
  set -- $spec
  echo "########## OPS $1  $(uptime | sed 's/.*load average/load average/') ##########"
  python3 perf/p3l1/fold_ab.py --arm "ops_$1" --l1-out "$2" --bias-l1-norm "$3" --l1-bw 1 \
    --instrument ops --repeat 0 --out "perf/p3l1/ops_$1.json" 2>&1 \
    | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning" \
    | tail -30
done
echo "########## POST DONE ##########"
