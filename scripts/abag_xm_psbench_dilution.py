"""Test the MECHANISM behind the dockq_wave offset, rather than just its sign.

The claim in the state doc is that dockq_wave sits above our per-interface label because it
averages in the near-rigid intra-antibody H-L interface, which is almost always modelled well.
That is a story until the H-L interface is actually measured. So score, on the same models the
leg-(i) pilot used: H-L (intra-antibody), H-A and L-A (the two Ab-Ag interfaces). If H-L is
near-perfect while H-A/L-A are poor, the mechanism is a measured fact.
"""
import json
import pathlib
import subprocess

import numpy as np

WT = pathlib.Path(__file__).resolve().parent.parent  # this checkout, not a torn-down slug worktree
PSB = pathlib.Path("/home/ttuser/abag_xm/psbench")
PM = PSB / "Multimer_7_2024_8_2025_dataset/Predicted_Models"
PY = "/home/ttuser/tt-bio/env/bin/python3"

TARGETS = {t["id"]: t for t in json.load(
    open(WT / "docs/implementation-parity-data/abag-xm-psbench-leg-i-targets.json"))}
pilot = json.load(open(PSB / "leg_i_pilot.json"))


def dockq(model, native, c1, c2):
    p = subprocess.run([PY, str(WT / "scripts/abag_xm_dockq_interface.py"),
                        str(model), str(native), c1, c2],
                       capture_output=True, text=True, cwd=str(WT),
                       env={"PYTHONPATH": str(WT), "PATH": "/usr/bin:/bin"})
    try:
        return json.loads(p.stdout).get("dockq")
    except Exception:
        return None


out = []
for tid in sorted({r["target"] for r in pilot}):
    meta = TARGETS[tid]
    native = PSB / "native" / f"{tid}.cif"
    rows = [r for r in pilot if r["target"] == tid and r.get("HA") is not None]
    print("\n%s  H=%s L=%s A=%s  (%d models)" % (tid, meta["H"], meta["L"], meta["A"], len(rows)))
    for r in rows:
        m = PM / tid / tid / r["model"]
        rec = dict(r)
        rec["HL"] = dockq(m, native, meta["H"], meta["L"])
        rec["LA"] = dockq(m, native, meta["L"], meta["A"])
        out.append(rec)
    for k in ("HL", "HA", "LA", "dockq_wave"):
        v = [x[k] for x in out if x["target"] == tid and x.get(k) is not None]
        if v:
            print("   %-11s n=%2d  median %.3f  min %.3f  max %.3f  >=0.23: %d/%d"
                  % (k, len(v), float(np.median(v)), min(v), max(v),
                     sum(1 for y in v if y >= 0.23), len(v)))

json.dump(out, open(PSB / "leg_i_dilution.json", "w"), indent=1)

print("\n============ POOLED MECHANISM CHECK (n=%d) ============" % len(out))
for k, desc in (("HL", "intra-antibody H-L  (predicted near-rigid, easy)"),
                ("HA", "Ab-Ag  H-A          (our primary label)"),
                ("LA", "Ab-Ag  L-A"),
                ("dockq_wave", "dockq_wave          (their all-interface average)")):
    v = [x[k] for x in out if x.get(k) is not None]
    print("  %-12s median %.3f   mean %.3f   >=0.23 in %d/%d"
          % (k, float(np.median(v)), float(np.mean(v)),
             sum(1 for y in v if y >= 0.23), len(v)))
hl = [x["HL"] for x in out if x.get("HL") is not None]
ha = [x["HA"] for x in out if x.get("HA") is not None]
print("\n  H-L median (%.3f) minus H-A median (%.3f) = %+.3f"
      % (float(np.median(hl)), float(np.median(ha)), float(np.median(hl) - np.median(ha))))
print("  If H-L is high and flat while H-A varies, the average is dominated by the easy")
print("  interface and dockq_wave cannot resolve Ab-Ag quality -- the mechanism, measured.")
