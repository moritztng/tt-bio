# Shared environment for every xmpage leg. Source it, do not copy it.
WT=/home/ttuser/.coworker/wt/protenix-opendde-softmax-perfpage-remeasure
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:protenix-opendde-softmax-perfpage-remeasure
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm BENCHLOCK_MAXLOAD=2.0
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
FIX=$WT/perf/size512/fixtures
O=$WT/perf/xmpage
