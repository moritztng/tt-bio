#!/bin/bash
# The card leg of the e2e->main merge: the 19 device suites card-free pytest cannot run, and
# two designs reproduced against graded CIFs. Card 3 on qb2, both interpreters for the suites.
#
#   nohup bash perf/rfd4_land/device.sh > ~/rfd4-main-runs/device.log 2>&1 &
set -u
WT=/home/ttuser/.coworker/wt/rfd4-main
R=/home/ttuser/rfd4-main-runs
V10=/home/ttuser/scratch/shipflow/venv-1384f4c1b7bf/bin/python
V12=/home/ttuser/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
mkdir -p $R
cd $WT || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:rfd4-main PYTHONPATH=$WT
SUITES="tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py tests/test_rfd4_ae.py tests/test_rfd4_io.py"

for leg in 310:$V10 312:$V12; do
  py=${leg%%:*}; P=${leg#*:}
  echo "=== device suites $py"
  timeout 3000 $P -m pytest -q -p no:cacheprovider --junitxml=$R/device$py.xml $SUITES > $R/device$py.log 2>&1
  echo "device$py rc=$?" >> $R/done
done

echo "=== unconditional design, CLI, reproducing rfd4-cond-protein seed 0"
timeout 2400 $V12 perf/rfd4_land/clockrun.py --out $R/runs.jsonl --tag uncond_cli -- \
  $V12 -m tt_bio.main design --model rfd4 \
  /home/ttuser/.coworker/wt/rfd4-cond-protein/perf/rfd4/cond/inputs/monomer-short.cif.zst \
  --out_dir $R/unconditional --seed 0 --num_designs 2 --recipe thinking_medium --annotated_cif
echo "uncond rc=$?" >> $R/done

echo "=== 128-token pdl1 binder, same harness and rng as rfd4-e2e s0"
timeout 2400 $V12 perf/rfd4/e2e/design.py --spec /home/ttuser/rfd4-inputs/pdl1-128.yaml \
  --weights /home/ttuser/rfd4-bundle --out $R/design128 --seed 0 --designs 2 --rng upstream \
  > $R/design128.log 2>&1
echo "design128 rc=$?" >> $R/done

echo "=== compares"
$V12 perf/rfd4_land/cif_compare.py $R/unconditional/monomer-short.cif_0_generated.cif \
  /home/ttuser/rfd4-cond-runs/unconditional/monomer-short.cif_0_generated.cif --json $R/cmp-uncond0.json
$V12 perf/rfd4_land/cif_compare.py $R/unconditional/monomer-short.cif_1_generated.cif \
  /home/ttuser/rfd4-cond-runs/unconditional/monomer-short.cif_1_generated.cif --json $R/cmp-uncond1.json
$V12 perf/rfd4_land/cif_compare.py $R/design128/pdl1_binder_128_s0_r1_generated.cif \
  /home/ttuser/rfd4-e2e-runs/design128/pdl1_binder_128_s0_r1_generated.cif --json $R/cmp-128.json
echo ALLDONE >> $R/done
