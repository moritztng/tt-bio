#!/usr/bin/env bash
# E2 -- does the SCOPED lever deliver what the global one measured? Three processes in ONE
# benchlock hold, the two incumbent legs bracketing the lever leg so the A/A is measured in the
# same window rather than asserted from another pass.
#
# Same shape as p10_m4_chain.sh, and deliberately so: p10's 122.980 s / 161.891 s were taken one
# process an arm, so a three-process replication is directly comparable and an interleaved one is
# not. The only thing that changed is where the arm comes from -- `--l1-padded-plan` on AF2's own
# blocks instead of TT_BIO_FP32_SOFTMAX_L1_PADDED on the module global. The env var stays UNSET
# for the whole run, so a leg that comes out fast because the box leaked the flag cannot happen.
#
#   BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=600 \
#     ~/.coworker/scripts/benchlock.sh flight-land-pxdesign-af2ig -- bash perf/pxdesign/e2_scoped_chain.sh
#
# benchlock exit 75 = never got the box. That is a DEFER, not a measurement taken anyway.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PDB="$WT/.e2/e2_848.pdb"
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs .e2

# The 848 fixture: the 768-residue target out of perf/pxdesign/targets/laczc_768.cif plus the
# anchor complex's 80-residue binder. Deterministic from two committed files, so it is rebuilt
# rather than carried. sha256 74790715066f4eabcabba4414e42c7a8b4c75d9d79365d1286d6dd0354980634.
# The interface is nonsense on purpose: every leg that reads this file is a timing leg.
if [ ! -f "$PDB" ]; then
  PYTHONPATH="$WT" "$PY" scripts/af2_port/complex_input.py \
      --target-residues 768 --binder-residues 80 --out "$PDB" >/dev/null || exit 1
fi
sha256sum "$PDB"

CARD="${E2_CARD:?set E2_CARD to the card grant this task holds}"
export TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:flight-land-pxdesign-af2ig
export PYTHONPATH="$WT"
# The fused triangle attention failed its accuracy gate at 2 of 50 decision flips. Both arms pin
# the materialised fp32-softmax path in every stack, so nothing here inherits it from the box.
unset TT_BIO_TRIATT_FUSED_HIFI
unset TT_BIO_FP32_SOFTMAX_L1_PADDED

echo "=== $(date -Is) start; loadavg $(cut -d\  -f1-3 /proc/loadavg) ==="
i=0
for arm in off on off; do
  i=$((i + 1))
  out="perf/pxdesign/tt_pxd_e2_fold_848_leg${i}_${arm}.json"
  echo "=== $(date -Is) leg $i/3  --l1-padded-plan $arm -> $out ==="
  timeout 2400 "$PY" scripts/af2_port/fold_timing.py \
      --pdb "$PDB" --arm device --reps 3 --recycles 3 --triatt-fused none \
      --l1-padded-plan "$arm" \
      --label "e2_848_l1padded_${arm}_leg${i}" --out "$out" \
      > "logs/e2_fold_leg${i}.out" 2> "logs/e2_fold_leg${i}.err"
  echo "rc=$?"
  tail -2 "logs/e2_fold_leg${i}.err"
done
echo "=== $(date -Is) done; loadavg $(cut -d\  -f1-3 /proc/loadavg) ==="
