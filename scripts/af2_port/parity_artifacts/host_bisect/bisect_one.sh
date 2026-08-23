# .p18/bisect_one.sh <commit> <card> [extra args...]
C=$1; CARD=$2; shift 2
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p18
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=$WT/.p18/trees/$C
rm -rf $D && mkdir -p $D
( cd $WT && git archive $C tt_bio scripts/af2_port ) | tar -x -C $D
cd $D
OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p18 PYTHONPATH=$D \
  $PY -u scripts/af2_port/tap_gate.py --device "$@" \
  > $WT/.p18/bisect/bs_$C.json 2> $WT/.p18/bisect/bs_$C.err
echo "exit $? $C card$CARD"
