# Shared environment for every oddedgx leg. Source it, do not copy it.
WT=/home/ttuser/.coworker/wt/opendde-beat-dgx-h200
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-beat-dgx-h200
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddedgx
