"""CA-lDDT vs 1HCL for the SHIPPED cond-hoist arm at 512 aa. Host-only, no device.

cdk2x2_512's whole-molecule RMSD cannot adjudicate this lever: base and hoist sit 11.8 A apart
on an unconstrained hinge while the fixture's own base-against-base seed floor spans
1.37-17.50 A. That is the exact screen b2z2-union-land proved blind -- it cleared Protenix-v2's
silu regression. lDDT is superposition-free and local, so a rigid inter-lobe hinge rotation
barely moves it, and it is scored against the experimental answer rather than against another
fold of the same model.
"""
import sys
from pathlib import Path
import numpy as np
R = Path("/home/ttuser/.coworker/wt/c13-land-first")
sys.path.insert(0, str(R / "perf" / "other512"))
sys.path.insert(0, str(R / "perf" / "fused_sdpa"))
from of3_score_ref import ca_map
from basin_lddt import lddt_per_residue

gt = ca_map(R / "perf" / "fused_sdpa" / "cifs" / "1hcl.cif")
D = R / "perf" / "c13_land" / "remeasure512_cifs"

def domains(cif):
    m = ca_map(cif)                      # single chain, seq ids 1..512
    return {1: {i: m[i] for i in range(1, 257) if i in m},
            2: {i - 256: m[i] for i in range(257, 513) if i in m}}

arms = {"base": D / "512_r0_p0_base.cif", "hoist": D / "512_r0_p2_hoist.cif",
        "silu": D / "512_r0_p1_silu.cif", "both": D / "512_r0_p3_both.cif",
        "base_AA": D / "512_r0_p4_base.cif"}
dom = {a: domains(p) for a, p in arms.items()}

print(f"1HCL CA residues: {len(gt)}")
res = {}
for d in (1, 2):
    keys = sorted(set(dom["base"][d]) & set(gt))
    bad = [k for k in keys if dom["base"][d][k][0] != gt[k][0]]
    keys = [k for k in keys if k not in bad]
    ref = np.array([gt[k][1] for k in keys])
    print(f"\n--- domain {d}: {len(keys)} CA scored ({len(bad)} residue-identity mismatches dropped)")
    for a in arms:
        X = np.array([dom[a][d][k][1] for k in keys])
        _, g = lddt_per_residue(X, ref)
        res[(a, d)] = g
        print(f"    {a:<8} CA-lDDT vs 1HCL {g:.5f}")
print("\n=== margin vs base, per domain (the b2z2 screen) ===")
for a in ("hoist", "silu", "both", "base_AA"):
    print(f"  {a:<8} " + "  ".join(f"d{d} {res[(a,d)] - res[('base',d)]:+.5f}" for d in (1, 2)))
