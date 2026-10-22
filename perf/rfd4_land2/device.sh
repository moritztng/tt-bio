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
#   TT_VISIBLE_DEVICES=<card> nohup bash perf/rfd4_land2/device.sh <tree> <outdir> [legs] > <log> 2>&1 &
#
# legs (default "310 312 design cli") picks which legs run, so a rerun repeats only what failed.
#
# Before the first leg it waits, up to 6 h, for the card to show no open fd for 3 checks in a row:
# release-gate legs take a card without booking it, and an open on top of one kills both runs.
set -u
WT=${1:?tree}
R=${2:?outdir}
LEGS=" ${3:-310 312 design cli} "
CARD=${TT_VISIBLE_DEVICES:?set TT_VISIBLE_DEVICES to the granted card}
V10=/home/ttuser/scratch/shipflow/venv-1384f4c1b7bf/bin/python
V12=/home/ttuser/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
MAINREF=/home/ttuser/rfd4-main-runs/design128/pdl1_binder_128_s0_r1_generated.cif
E2EREF=/home/ttuser/rfd4-e2e-runs/design128/pdl1_binder_128_s0_r1_generated.cif
mkdir -p "$R"
clear=0
for _ in $(seq 1080); do
  if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q "/dev/tenstorrent/$CARD\$"; then clear=0; else clear=$((clear+1)); fi
  [ $clear -ge 3 ] && break
  sleep 20
done
[ $clear -ge 3 ] || { echo "card $CARD never came free" >> "$R/done"; exit 1; }
echo "card $CARD free $(date -u +%FT%TZ)" >> "$R/done"
cd "$WT" || exit 1
export TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:rfd4-land2 PYTHONPATH=$WT
SUITES="tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py tests/test_rfd4_ae.py tests/test_rfd4_io.py"

for leg in 310:$V10 312:$V12; do
  py=${leg%%:*}; P=${leg#*:}
  [[ $LEGS == *" $py "* ]] || continue
  echo "=== device suites $py"
  timeout 3000 $P -m pytest -q -p no:cacheprovider --junitxml=$R/device$py.xml $SUITES > $R/device$py.log 2>&1
  echo "device$py rc=$?" >> $R/done
done

if [[ $LEGS == *" design "* ]]; then
echo "=== 128-token pdl1 binder, rfd4-e2e's harness and rng at seed 0"
timeout 2400 $V12 perf/rfd4/e2e/design.py --spec /home/ttuser/rfd4-inputs/pdl1-128.yaml \
  --weights /home/ttuser/rfd4-bundle --out $R/design128 --seed 0 --designs 2 --rng upstream \
  > $R/design128.log 2>&1
echo "design128 rc=$?" >> $R/done

MINE=$R/design128/pdl1_binder_128_s0_r1_generated.cif
$V12 perf/rfd4_land/cif_compare.py "$MINE" "$MAINREF" --json $R/cmp-128-vs-main.json
$V12 perf/rfd4_land/cif_compare.py "$MINE" "$E2EREF" --json $R/cmp-128-vs-e2e.json
fi

# The same 128-token spec through the user path, `tt-bio design --model rfd4`, on this tree and on
# main (MAIN_TREE, a checkout of origin/main) at one seed: the designs must be the same atoms.
if [[ $LEGS == *" cli "* ]]; then
MAIN_TREE=${MAIN_TREE:-/home/ttuser/scratch/rfd4-land2/mainref}
for leg in land:$WT main:$MAIN_TREE; do
  name=${leg%%:*}; T=${leg#*:}
  (cd "$T" && PYTHONPATH=$T timeout 1500 $V12 -m tt_bio.main design --model rfd4 \
     /home/ttuser/rfd4-inputs/pdl1-128.yaml --out_dir $R/cli_$name --seed 0 --num_designs 1) \
     > $R/cli_$name.log 2>&1
  echo "cli_$name rc=$?" >> $R/done
done
for f in $(cd $R/cli_main && find . -name "*.cif" | sort); do
  $V12 perf/rfd4_land/cif_compare.py "$R/cli_land/$f" "$R/cli_main/$f" --json "$R/cmp-cli-$(basename $f).json"
done
fi
echo ALLDONE >> $R/done
