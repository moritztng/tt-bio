#!/bin/bash
# M18's confident-target accuracy leg: 7ROA at 117 aa, lever off then on, scored by the release
# gate's own CA-RMSD/TM against the deposited structure.
#
# WHY THIS TARGET. Every accuracy number M18 has so far is on cdk2x2, a size-ladder PERF fixture
# where OpenFold3 never exceeds pLDDT 0.55 and the 512 aa variant is a tandem duplicate whose halves
# hinge freely between seeds (CA seed floor 22.5 A). On 7ROA the model is 1.775 A / TM 0.890 from
# the experimental structure, so a lever can be scored against an answer rather than against another
# run of itself.
#
# THE PRECEDENT, and it is exact. release_gate.py records RF3 flipping to this same fused-SDPA
# default and landing on its pre-flip number: 1.238 -> 1.239 A, TM 0.958 both sides, same card same
# grid. If M18 lands on 1.775 A the same way, the 298 aa failure is a property of low-confidence
# fixtures; if it degrades, the lever is closed on accuracy for good.
#
# READ IT AS: off-arm RMSD/TM against on-arm RMSD/TM, both against ground truth. The gate prints
# both. This is NOT a timing run -- do not quote seconds from it; the gate is not benchlocked here.
set -u
WT=/home/ttuser/.coworker/wt/allm-gates
cd "$WT" || exit 1
CARD="${1:?usage: m18_7roa_accuracy.sh <card>}"
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-gates
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/allm_gates/m18_7roa
mkdir -p "$OUT"

for ARM in off on; do
  # `-openfold3.trunk` forces the site OFF, a bare token forces it ON -- the `_site_flag` grammar,
  # so neither arm depends on what the shipped default happens to be when this is run.
  if [ "$ARM" = on ]; then export TT_BIO_TRIATT_SDPA_HIFI_AB="openfold3.trunk"
  else                     export TT_BIO_TRIATT_SDPA_HIFI_AB="-openfold3.trunk"; fi
  echo "=== 7ROA arm=$ARM  TT_BIO_TRIATT_SDPA_HIFI_AB=$TT_BIO_TRIATT_SDPA_HIFI_AB $(date -u +%FT%TZ) ==="
  # --load-ceiling is raised deliberately. This leg scores CA-RMSD/TM against the DEPOSITED
  # structure, which host load cannot move; the gate's ceiling exists to protect TIMED runs
  # and no second is quoted from here. Refusing to score accuracy because a co-tenant is
  # busy would leave the lever undecided for a reason unrelated to the measurement.
  $PY -u scripts/release_gate.py --model openfold3 --keep \
      --load-ceiling "${M18_LOAD_CEILING:-128}" 2>&1 | tee "$OUT/gate_$ARM.log"
  echo "rc=${PIPESTATUS[0]} arm=$ARM"
done

echo
echo "=== RMSD / TM per arm (gate's own scorer, vs the deposited structure) ==="
for ARM in off on; do
  printf '%-4s ' "$ARM"
  grep -oE "RMSD [0-9.]+ A?.*TM [0-9.]+|rmsd[= ][0-9.]+.*tm[= ][0-9.]+" "$OUT/gate_$ARM.log" | tail -1
done
echo
echo "Reference on record: OpenFold3 7ROA = 1.775 A / TM 0.890 (release_gate.py MODELS)."
echo "RF3 across the identical flip: 1.238 -> 1.239 A, TM 0.958 both sides."
