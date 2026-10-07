#!/bin/bash
# The card-free test of the dropout finding.
#
# tt-bio's on-card Evoformer applies no dropout: "dropout" appears zero times in
# tt_bio/bindcraft2.py and tt_bio/tenstorrent.py, while BindCraft 2 threads batch["use_dropout"]
# into every Evoformer sub-layer. design_dropout defaults to true, and the stage plan turns it off
# for harden alone. So the on-card arm runs dropout-free for all 120 rounds of screen+refine+anneal
# and the two arms coincide only at harden.
#
# Arm ON  is the host-JAX control (what the reporter's trunk="jax" arm does).
# Arm OFF is what the device arm effectively does, reproduced on the host with no card.
# If OFF holds a strong anneal and then falls at harden where ON holds, the missing dropout alone
# makes the signature in issue #17.
set -u
WT=/home/ttuser/.coworker/wt/bci-accept
PY=/home/ttuser/fdv_fresh/venv/bin/python
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:$WT
cd "$WT" || exit 1
TRAJ=${1:-3}
LEN=${2:-60}

for arm in true false; do
  project=/tmp/bci17_dropout_$arm
  rm -rf "$project"
  echo "=== arm design_dropout=$arm, $TRAJ trajectories, binder $LEN ==="
  nice -n 10 timeout 28800 "$PY" perf/bci_accept/capture_logits.py \
    --target-pdb /tmp/4zqk.pdb --af2-weights /home/ttuser/bcx_e2e/af2_params \
    --out "$WT/.bci/logits_$arm.npz" --dump-states "$WT/.bci/harden_states_$arm.pkl" \
    --project "$project" --design-dropout "$arm" --trajectories "$TRAJ" \
    --binder-lengths "$LEN" "$LEN"
  echo "=== arm $arm rc=$? ==="
done

echo "=== per-stage iptm, both arms ==="
for arm in true false; do
  for csv in /tmp/bci17_dropout_$arm/1_Trajectories/*/*_losses.csv; do
    [ -f "$csv" ] || continue
    echo "-- dropout=$arm $(basename "$(dirname "$csv")")"
    "$PY" - "$csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
iptm = next((k for k in rows[0] if k.endswith('.iptm')), None)
ptm = next((k for k in rows[0] if k.endswith('.ptm')), None)
last = {}
for r in rows:
    last[r['phase']] = r
for phase in ('screen', 'refine', 'anneal', 'harden', 'mutate'):
    if phase in last:
        print(f"   {phase:8s} iptm={float(last[phase][iptm]):.3f} ptm={float(last[phase][ptm]):.3f}")
PYEOF
  done
done
echo "=== dropout arms done ==="
