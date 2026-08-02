#!/bin/bash
# p2 step 2b: diffusion numerics under 0.75 vs 0.68 vs committed fixtures.
# Waits for the step-1 RFD3 warmups to free cards 1+2, then folds:
#   card 1: boltz2-68, then protenix-68
#   card 2: protenix-75
#   (boltz2-75 ran separately on card 3 — already done)
# Then scores: fixture-seed0 vs each version, 0.75-vs-0.68 cross, and CA PCC/maxabs.
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
NUM=/tmp/ttnn075-numerics
V68=/home/ttuser/.coworker/scout-venvs/v68/bin
V75=/home/ttuser/.coworker/scout-venvs/v75/bin
mkdir -p $NUM
LOG=/tmp/step2b_driver.log
echo "driver restart $(date -u)" >> $LOG

STAGE=$($V75/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from pathlib import Path
import full_parity_gate as G
leg = [l for l in G.LEGS if l.id == "protenix-prot-msa"][0]
msa_dir, args = G.stage_msa(leg, Path("/tmp/ptx_msa_work"))
print(" ".join(args))
PY
)
echo "protenix msa args: $STAGE" >> $LOG

# wait for step-1 warmups (frees cards 1+2)
while ! grep -q "warmups done" /tmp/step1_driver.log 2>/dev/null; do sleep 60; done
echo "warmups done observed $(date -u)" >> $LOG

(
  SCOUT_CARD=1 scripts/scout_run_leg.sh 68 $V68/tt-bio \
    predict examples/trpcage_no_msa.yaml --model boltz2 --out_dir $NUM/boltz2_68/seed0 --override \
    --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --single_sequence \
    >/tmp/b2_68.log 2>&1
  echo "boltz2_68 rc=$? $(date -u)" >> $LOG
  SCOUT_CARD=1 scripts/scout_run_leg.sh 68 $V68/tt-bio \
    predict examples/prot.yaml --model protenix-v2 --out_dir $NUM/protenix_68/seed0 --override \
    --seed 0 --sampling_steps 200 --diffusion_samples 5 $STAGE \
    >/tmp/ptx_68.log 2>&1
  echo "protenix_68 rc=$? $(date -u)" >> $LOG
) &

(
  SCOUT_CARD=2 scripts/scout_run_leg.sh 75 $V75/tt-bio \
    predict examples/prot.yaml --model protenix-v2 --out_dir $NUM/protenix_75/seed0 --override \
    --seed 0 --sampling_steps 200 --diffusion_samples 5 $STAGE \
    >/tmp/ptx_75.log 2>&1
  echo "protenix_75 rc=$? $(date -u)" >> $LOG
) &

wait
echo "all folds done $(date -u)" >> $LOG

$V75/python - <<'PY' >> $LOG 2>&1
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
import full_parity_gate as G
import subprocess
import numpy as np
import gemmi

NUM = Path("/tmp/ttnn075-numerics")
V75 = "/home/ttuser/.coworker/scout-venvs/v75/bin/python"

def results_dir(d):
    rd = G._find_results_dir(Path(d))
    return str(rd) if rd else None

def ca_coords(cif):
    st = gemmi.read_structure(str(cif))
    out = []
    for m in st:
        for ch in m:
            for r in ch:
                for a in r:
                    if a.name == "CA":
                        out.append(list(a.pos))
    return np.array(out)

def all_coords(cif):
    st = gemmi.read_structure(str(cif))
    out = []
    for m in st:
        for ch in m:
            for r in ch:
                for a in r:
                    out.append(list(a.pos))
    return np.array(out)

def pcc(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

jobs = [
    ("boltz2-trpcage-nomsa", "boltz2/trpcage/nomsa_200step_1sample_3recycle_bf16",
     "trpcage_no_msa", "boltz2_68", "boltz2_75"),
    ("protenix-prot-msa", "protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16",
     "prot", "protenix_68", "protenix_75"),
]
summary = {}
for leg_id, fixture, tid, d68, d75 in jobs:
    r68, r75 = results_dir(NUM/d68), results_dir(NUM/d75)
    fix_seed0 = str(G._fixture_dir(fixture) / "seed0")
    print(f"{leg_id}: dir68={r68} dir75={r75}", flush=True)
    entry = {}
    for tag, rd in [("68", r68), ("75", r75)]:
        if not rd:
            entry[tag] = "MISSING"
            continue
        out = NUM / f"score_{leg_id}_{tag}.json"
        subprocess.run([V75, "scripts/pharma_parity.py", "structures",
                        "--ref-dirs", fix_seed0, "--dev-dirs", rd,
                        "--label", f"{leg_id}-ttnn{tag}-vs-fixture", "--out", str(out)])
        entry[f"{tag}_vs_fixture"] = json.loads(out.read_text()) if out.exists() else "score failed"
    if r68 and r75:
        out = NUM / f"score_{leg_id}_75vs68.json"
        subprocess.run([V75, "scripts/pharma_parity.py", "structures",
                        "--ref-dirs", r68, "--dev-dirs", r75,
                        "--label", f"{leg_id}-75vs68", "--out", str(out)])
        entry["75vs68_structures"] = json.loads(out.read_text()) if out.exists() else "score failed"
        c68 = Path(r68) / "structures" / f"{tid}.cif"
        c75 = Path(r75) / "structures" / f"{tid}.cif"
        if c68.exists() and c75.exists():
            ca68, ca75 = ca_coords(c68), ca_coords(c75)
            al68, al75 = all_coords(c68), all_coords(c75)
            entry["75vs68_direct"] = {
                "n_ca": int(len(ca68)), "n_atoms": int(len(al68)),
                "ca_pcc": pcc(ca68, ca75) if len(ca68) == len(ca75) else "len mismatch",
                "ca_maxabs_A": float(np.abs(ca68 - ca75).max()) if len(ca68) == len(ca75) else None,
                "allatom_pcc": pcc(al68, al75) if len(al68) == len(al75) else "len mismatch",
                "allatom_maxabs_A": float(np.abs(al68 - al75).max()) if len(al68) == len(al75) else None,
            }
    summary[leg_id] = entry

(NUM / "numerics_summary.json").write_text(json.dumps(summary, indent=2))
print("SCORING DONE", flush=True)
PY
echo "scoring rc=$? $(date -u)" >> $LOG
touch /tmp/step2b_DONE
