#!/bin/bash
# The one thing TT_BIO_APB_CONCAT_HEADS still owes on the accuracy axis.
#
# Pass 19 proved the flag reaches protenix-v2 (its l1-budget md5 moves when the flag moves), and
# its Angstrom evidence is boltz-2 only. The gate's protenix-v2 floor is 6.0 A against a measured
# 3.87 A -- wide enough that the silu regression Moritz refused in September would have passed it
# -- so a green gate is not an accuracy reading.
#
# Five folds of the gate's own protenix-v2 target at the gate's own protocol (200 sampling steps,
# 1 diffusion sample, single-sequence), so this is scored on what the gate already folds:
#   off seed 0 run a   the reference arm
#   on  seed 0 run a   the signal: paired, same seed, flag the only difference
#   off seed 0 run b   the A/A control -- must be 0.000000 A or nothing else here is readable
#   off seed 1 / 2     the seed-scatter floor the signal has to be judged against
#
# Angstrom, not seconds: no timing guard is taken, none is needed, and no clock is claimed.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-1}
OUT=$WT/perf/c14_land/apb_pvx_cifs
cd "$WT" || exit 1
mkdir -p "$OUT"

one() {  # arm seed runlabel
  local arm=$1 seed=$2 run=$3
  local work=$WT/perf/c14_land/pvx_scratch_${arm}_${seed}_${run}
  rm -rf "$work"; mkdir -p "$work"
  env TT_BIO_APB_CONCAT_HEADS=$([ "$arm" = on ] && echo 1 || echo 0) \
      TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      "$PY" -m tt_bio.main predict examples/affinity_fkg.yaml --model protenix-v2 \
        --single_sequence --sampling_steps 200 --diffusion_samples 1 --seed "$seed" \
        --out_dir "$work" > "$WT/perf/c14_land/pvx_${arm}_${seed}_${run}.log" 2>&1
  local rc=$?
  local d=$OUT/fkg_${arm}${seed}_${run}
  mkdir -p "$d"
  find "$work" -name '*.cif' -exec cp {} "$d"/ \;
  echo "$(date -u +%H:%M:%SZ) arm=$arm seed=$seed run=$run rc=$rc cifs=$(ls "$d" | wc -l)"
  rm -rf "$work"
}

one off 0 a
one on  0 a
one off 0 b
one off 1 a
one off 2 a
echo ALL LEGS DONE
