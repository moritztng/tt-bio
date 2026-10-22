#!/bin/bash
# The card leg of the land2 -> main merge: the device suites card-free pytest cannot run, and the
# 128-token pdl1 binder reproduced on the merged tree.
#
# The reference is rfd4-main's own run of that design on the tree that is now main
# (~/rfd4-main-runs/design128/..._s0_r1_generated.cif, venv312, qb2 card 3, AICLK median 1350).
# Same harness, same rng, same venv, so this branch has to land on it exactly: everything this
# branch adds to the design path is either new surface nothing default calls, or the trimul work
# rfd4-perf-tri graded bit-identical. A nonzero deviation here is a regression, not a venv.
# cmp-128-vs-e2e.json is the older upstream-paired reference beside it, for continuity with
# LAND.md's 0.221 A (which attribution showed is the venv, not the merge).
#
#   TT_VISIBLE_DEVICES=<card> nohup bash perf/rfd4_land2/device.sh <tree> <outdir> > <log> 2>&1 &
set -u
WT=${1:?tree}
R=${2:?outdir}
CARD=${TT_VISIBLE_DEVICES:?set TT_VISIBLE_DEVICES to the granted card}
V10=/home/ttuser/scratch/shipflow/venv-1384f4c1b7bf/bin/python
V12=/home/ttuser/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
MAINREF=/home/ttuser/rfd4-main-runs/design128/pdl1_binder_128_s0_r1_generated.cif
E2EREF=/home/ttuser/rfd4-e2e-runs/design128/pdl1_binder_128_s0_r1_generated.cif
mkdir -p "$R"
cd "$WT" || exit 1
export TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:rfd4-land2 PYTHONPATH=$WT
SUITES="tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py tests/test_rfd4_ae.py tests/test_rfd4_io.py"

for leg in 310:$V10 312:$V12; do
  py=${leg%%:*}; P=${leg#*:}
  echo "=== device suites $py"
  timeout 3000 $P -m pytest -q -p no:cacheprovider --junitxml=$R/device$py.xml $SUITES > $R/device$py.log 2>&1
  echo "device$py rc=$?" >> $R/done
done

echo "=== 128-token pdl1 binder, rfd4-e2e's harness and rng at seed 0"
timeout 2400 $V12 perf/rfd4/e2e/design.py --spec /home/ttuser/rfd4-inputs/pdl1-128.yaml \
  --weights /home/ttuser/rfd4-bundle --out $R/design128 --seed 0 --designs 2 --rng upstream \
  > $R/design128.log 2>&1
echo "design128 rc=$?" >> $R/done

MINE=$R/design128/pdl1_binder_128_s0_r1_generated.cif
$V12 perf/rfd4_land/cif_compare.py "$MINE" "$MAINREF" --json $R/cmp-128-vs-main.json
$V12 perf/rfd4_land/cif_compare.py "$MINE" "$E2EREF" --json $R/cmp-128-vs-e2e.json
echo ALLDONE >> $R/done
