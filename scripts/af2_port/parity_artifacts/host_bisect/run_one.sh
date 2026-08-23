# .p18/run_one.sh <card> <tag> [extra tap_gate args...]
CARD=$1; TAG=$2; shift 2
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p18
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT
OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p18 PYTHONPATH=$WT \
  $PY -u scripts/af2_port/tap_gate.py --device "$@" \
  > $WT/.p18/bisect/$TAG.json 2> $WT/.p18/bisect/$TAG.err
echo "exit $? $TAG"
