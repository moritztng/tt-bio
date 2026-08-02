"""p2 step 2b scoring, standalone: fixture-seed0 vs each version, 0.75-vs-0.68 cross,
plus direct CA/all-atom PCC+maxabs between the two device outputs."""
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
import subprocess
import numpy as np
import gemmi

NUM = Path("/tmp/ttnn075-numerics")
V75 = "/home/ttuser/.coworker/scout-venvs/v75/bin/python"
FIX = Path("docs/implementation-parity-data/ref-fixtures")


def results_dir(d):
    d = Path(d)
    hits = sorted(d.rglob("results.json"))
    return str(hits[0].parent) if hits else None


def coords(cif, ca_only):
    st = gemmi.read_structure(str(cif))
    out = []
    for m in st:
        for ch in m:
            for r in ch:
                for a in r:
                    if not ca_only or a.name == "CA":
                        out.append(list(a.pos))
    return np.array(out)


def pcc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


jobs = [
    ("boltz2-trpcage-nomsa", "boltz2/trpcage/nomsa_200step_1sample_3recycle_bf16",
     "trpcage_no_msa", "boltz2_68", "boltz2_75"),
    ("protenix-prot-msa", "protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16",
     "prot", "protenix_68", "protenix_75"),
]
summary = {}
for leg_id, fixture, tid, d68, d75 in jobs:
    r68, r75 = results_dir(NUM / d68), results_dir(NUM / d75)
    fix_seed0 = str(FIX / fixture / "seed0")
    print(f"{leg_id}: dir68={r68} dir75={r75}", flush=True)
    entry = {"dir68": r68, "dir75": r75}
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
            ca68, ca75 = coords(c68, True), coords(c75, True)
            al68, al75 = coords(c68, False), coords(c75, False)
            entry["75vs68_direct"] = {
                "n_ca": int(len(ca68)), "n_atoms": int(len(al68)),
                "ca_pcc": pcc(ca68, ca75) if ca68.shape == ca75.shape else "shape mismatch",
                "ca_maxabs_A": float(np.abs(ca68 - ca75).max()) if ca68.shape == ca75.shape else None,
                "allatom_pcc": pcc(al68, al75) if al68.shape == al75.shape else "shape mismatch",
                "allatom_maxabs_A": float(np.abs(al68 - al75).max()) if al68.shape == al75.shape else None,
            }
    summary[leg_id] = entry

(NUM / "numerics_summary.json").write_text(json.dumps(summary, indent=2))
print("SCORING DONE", flush=True)
