#!/usr/bin/env python3
"""The denominator floor (A14) and the 3e control for the diffusion-boundary comparison.

A14: a per-tensor relative L2 divides by the reference's own norm, so a tensor whose reference
gradient is numerically nothing reports an enormous relative error that means nothing -- another
row published a "1.142e+13 relative error" from a ref_norm of 1.4e-19. A floor has to be declared
BEFORE the worst case is read, and it has to be declared in the reference's units.

3e: the control must break what the check READS. The check reads a per-tensor relative L2 against
`grads_f64_r0.pt`'s `diffusion_module.*` entries, so the control replaces the compared side with
something known-wrong and shows the number moves. A14/ii: size it against the MEASURED baseline,
not a fixed 1 %, because the protocol's own 1 % scaling once failed to fire.

Reads only saved tensors. No forward.
"""
import json
import sys
import torch

BOUND = "/home/ttuser/of3t_diffusion_cap/diffusion_boundary.pt"
REF = "/home/ttuser/of3t/bundle_min/grads_f64_r0.pt"
OUT = "perf/of3t_diffusion/gradient_floor_control.json"

b = torch.load(BOUND, map_location="cpu", weights_only=False)
ours = b["grad_f64"]
ref = torch.load(REF, map_location="cpu", weights_only=False)

pairs = []
for n, g in ours.items():
    r = ref.get("diffusion_module." + n)
    if g is None or r is None:
        continue
    pairs.append((n, g.double(), r.double()))

def rel(x, y):
    return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

ref_norms = torch.tensor([float(torch.linalg.vector_norm(r)) for _, _, r in pairs])
tot_sq = float(sum(float(r.pow(2).sum()) for _, _, r in pairs))

# The floor is declared in the reference's own units and against the set's own scale: a tensor
# holding less than 1e-12 of the compared squared norm cannot move a conclusion about the whole,
# and its relative error is dominated by its own cancellation. Stated before the worst is read.
FLOOR_SHARE = 1e-12
floor_sq = FLOOR_SHARE * tot_sq

rep = {"floor": {"rule": "a tensor is compared only if its own squared gradient norm is at least "
                         "1e-12 of the compared set's squared norm",
                 "share": FLOOR_SHARE, "abs_sq": floor_sq,
                 "abs_norm": floor_sq ** 0.5,
                 "declared": "before the worst case below was read"},
       "compared_set": {"n_tensors": len(pairs), "sq_norm": tot_sq,
                        "ref_norm_min": float(ref_norms.min()),
                        "ref_norm_max": float(ref_norms.max())}}

kept, dropped = [], []
for n, g, r in pairs:
    (kept if float(r.pow(2).sum()) >= floor_sq else dropped).append((n, g, r))

def summarise(rows):
    vals = sorted((rel(g, r), n) for n, g, r in rows)
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": vals[len(vals) // 2][0],
            "worst": vals[-1][0], "worst_tensor": "diffusion_module." + vals[-1][1],
            "over_5e-2": sum(1 for v, _ in vals if v > 5.0e-2),
            "sq_norm_share_over_5e-2": sum(float(r.pow(2).sum()) for n, g, r in rows
                                           if rel(g, r) > 5.0e-2) / tot_sq}

rep["above_floor"] = summarise(kept)
rep["below_floor"] = {"n": len(dropped),
                      "tensors": ["diffusion_module." + n for n, _, _ in dropped][:12]}
print("above floor:", json.dumps(rep["above_floor"], indent=1))
print("below floor:", len(dropped))

base_median = rep["above_floor"]["median"]

# ---- 3e control, three arms, each breaking what the check reads -----------------------------
arms = {}

# zero arm: what the check reads if our side produced nothing. Relative L2 is exactly 1 per
# tensor, so a check that cannot separate this from a real comparison has no resolving power.
z = [(n, torch.zeros_like(g), r) for n, g, r in kept]
arms["zero_model"] = summarise(z)

# shuffle arm: same tensors, wrong pairing, within matching shapes. This is the arm that catches
# a comparison which is really only checking that both sides are "gradient-shaped".
byshape = {}
for n, g, r in kept:
    byshape.setdefault(tuple(g.shape), []).append((n, g, r))
sh = []
for shape, rows in byshape.items():
    if len(rows) < 2:
        continue
    for i, (n, g, r) in enumerate(rows):
        sh.append((n, rows[(i + 1) % len(rows)][1], r))
arms["shuffled_within_shape"] = summarise(sh)

# scaled arm, sized against the MEASURED baseline rather than a fixed 1 %: perturb our side by a
# relative epsilon equal to the measured median disagreement and confirm the check moves by it.
eps = base_median
gen = torch.Generator().manual_seed(20260919)
pert = []
for n, g, r in kept:
    d = torch.randn(g.shape, generator=gen, dtype=torch.float64)
    d = d / (torch.linalg.vector_norm(d) + 1e-300) * torch.linalg.vector_norm(r) * eps
    pert.append((n, g + d, r))
arms["perturbed_at_measured_median"] = {"epsilon": eps, **summarise(pert)}

rep["control"] = arms
for k, v in arms.items():
    print(f"  {k}: median {v.get('median'):.4e} worst {v.get('worst'):.4e} "
          f"over-bar {v.get('over_5e-2')}/{v.get('n')}")

json.dump(rep, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote", OUT)
