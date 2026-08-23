#!/usr/bin/env bash
# M4 -- the 848 fold, measured instead of projected. Three processes in ONE benchlock hold, the
# two incumbent legs bracketing the lever leg so the A/A is measured in the same window rather
# than asserted from another pass.
#
# One process an arm, because TT_BIO_FP32_SOFTMAX_L1_PADDED is read at import.
#
#   BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=600 \
#     ~/.coworker/scripts/benchlock.sh pxdesign-perf-p10 -- bash perf/pxdesign/p10_m4_chain.sh
#
# benchlock exit 75 = never got the box. That is a DEFER, not a measurement taken anyway.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-perf-p10
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PDB="$WT/.p10/p10_848.pdb"
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs .p10

# The 848 fixture: the 768-residue target out of perf/pxdesign/targets/laczc_768.cif plus the
# anchor complex.s 80-residue binder. Deterministic from two committed files, so it is rebuilt
# rather than carried. sha256 74790715066f4eabcabba4414e42c7a8b4c75d9d79365d1286d6dd0354980634.
# The interface is nonsense on purpose: every leg that reads this file is a timing leg.
if [ ! -f "$PDB" ]; then
  PYTHONPATH="$WT" "$PY" scripts/af2_port/complex_input.py \
      --target-residues 768 --binder-residues 80 --out "$PDB" >/dev/null || exit 1
fi
sha256sum "$PDB"

CARD="${P10_CARD:-0}"
export TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p10
export PYTHONPATH="$WT"
# The fused triangle attention failed its accuracy gate at 2 of 50 decision flips. Both arms pin
# the materialised fp32-softmax path in every stack, so nothing here inherits it from the box.
unset TT_BIO_TRIATT_FUSED_HIFI

echo "=== $(date -Is) start; loadavg $(cut -d\  -f1-3 /proc/loadavg) ==="
i=0
for arm in 0 1 0; do
  i=$((i + 1))
  out="perf/pxdesign/tt_pxd_p10_fold_848_leg${i}_arm${arm}.json"
  echo "=== $(date -Is) leg $i/3  TT_BIO_FP32_SOFTMAX_L1_PADDED=$arm -> $out ==="
  TT_BIO_FP32_SOFTMAX_L1_PADDED=$arm timeout 2400 "$PY" scripts/af2_port/fold_timing.py \
      --pdb "$PDB" --arm device --reps 3 --recycles 3 --triatt-fused none \
      --label "p10_848_l1padded${arm}_leg${i}" --out "$out" \
      > "logs/p10_fold_leg${i}.out" 2> "logs/p10_fold_leg${i}.err"
  echo "rc=$?"
  tail -2 "logs/p10_fold_leg${i}.err"
done
echo "=== $(date -Is) done; loadavg $(cut -d\  -f1-3 /proc/loadavg) ==="
