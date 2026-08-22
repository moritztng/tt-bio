#!/bin/bash
# .p19_leg.sh <tag> <card> <patch|none> [extra tap_gate args...]
TAG=$1; CARD=$2; PATCH=$3; shift 3
WT=/home/ttuser/.coworker/wt/pxdesign-af2ig-port-p19
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=$WT/.p19/trees/$TAG
mkdir -p $WT/.p19/out
rm -rf $D && mkdir -p $D
( cd $WT && git archive HEAD tt_bio scripts/af2_port ) | tar -x -C $D
if [ "$PATCH" != "none" ]; then
  $PY $WT/scripts/af2_port/parity_artifacts/host_bisect/leg_patch.py "$D/tt_bio/tenstorrent.py" "$PATCH" || { echo "PATCH FAILED"; exit 2; }
fi
cd $D
OMP_NUM_THREADS=8 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:pxdesign-af2ig-port-p19 PYTHONPATH=$D \
  $PY -u scripts/af2_port/tap_gate.py --device "$@" \
  > $WT/.p19/out/$TAG.json 2> $WT/.p19/out/$TAG.err
echo "exit $? tag=$TAG card=$CARD patch=$PATCH args=$*"
