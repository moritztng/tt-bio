"""Padding exposure of the two A18 forwards whose producing artifact records no masked variant:
the diffusion device arm and diffusion_conditioning. Both captures live on qb2; this reads them
on CPU only (no device is opened).

For each compared tensor it reports the REFERENCE's mass inside the real block, which is what
decides the direction of the bias: if the reference is exactly zero outside the real block, the
padded denominator equals the real one, so the padded reading cannot be diluted -- our own padded
entries can only add to the numerator, making the padded figure an UPPER bound on the real one.
If the reference carries mass outside, the reading is exposed to D95 and the size of the
exposure is 1 - q.
"""
import json
import sys

import torch

out = {"host": "qb2", "device_opened": False}


def block(t, m, axes):
    """select the real block of t along `axes` using bool mask m."""
    for a in sorted(axes, reverse=True):
        t = t.index_select(a, m.nonzero().flatten())
    return t


def exposure(name, ref, mask, axes):
    ref = ref.double()
    r = block(ref, mask, axes)
    n_all, n_real = float(ref.norm()), float(r.norm())
    outside_sq = max(n_all ** 2 - n_real ** 2, 0.0)
    return {"tensor": name, "shape": list(ref.shape), "axes_masked": axes,
            "ref_norm_all": n_all, "ref_norm_real_block": n_real,
            "q_ref_mass_inside_real_block": (n_real ** 2 / n_all ** 2) if n_all > 0 else None,
            "ref_sq_mass_outside": outside_sq,
            "ref_is_exactly_zero_outside": outside_sq == 0.0}


# ---- diffusion device arm (D30's 8.34e-03) ------------------------------------------------
dif = torch.load("/home/ttuser/of3t_diffusion_cap/diffusion_boundary.pt",
                 map_location="cpu", weights_only=False)
kw = dif["kwargs"] if "kwargs" in dif else dif
am = kw["atom_mask"].reshape(-1)
tm = kw["token_mask"].reshape(-1)
xl_out = dif["out"] if "out" in dif else dif.get("xl_out")
out["diffusion_device_arm"] = {
    "capture": "/home/ttuser/of3t_diffusion_cap/diffusion_boundary.pt",
    "keys": sorted(k for k in dif.keys()) if isinstance(dif, dict) else None,
    "compared_tensor": "xl_out, the diffusion module's atom positions",
    "compared_shape": list(xl_out.shape),
    "atom_mask_len": int(am.numel()), "atom_mask_sum": float(am.sum()),
    "token_mask_len": int(tm.numel()), "token_mask_sum": float(tm.sum()),
    "real_over_total_atoms": float(am.sum()) / int(am.numel()),
}
out["diffusion_device_arm"].update(
    exposure("xl_out[0,0]", xl_out[0, 0], am.bool(), [0]))
del dif, xl_out

# ---- diffusion_conditioning (A18 si / zij) -------------------------------------------------
for tag, path in (("base", "/home/ttuser/of3t_cond_cap/cond_boundary.pt"),
                  ("conditioned", "/home/ttuser/of3t_cond_cap/cond_boundary_cond.pt")):
    B = torch.load(path, map_location="cpu", weights_only=False)
    si_ref, zij_ref = B["si_ref"], B["zij_ref"]
    tok = B["token_mask"].reshape(-1).bool()
    rec = {"capture": path, "tokens": int(tok.numel()), "real_tokens": int(tok.sum()),
           "real_over_total_tokens": float(tok.sum()) / int(tok.numel()),
           "use_conditioning": bool(B["use_conditioning"]),
           "si": exposure("si_ref[0,:] over all noise levels", si_ref[0], tok, [1]),
           "si_per_sample_min_q": min(
               exposure("si_ref[0,k]", si_ref[0, k], tok, [0])["q_ref_mass_inside_real_block"]
               for k in range(si_ref.shape[1])),
           "zij": exposure("zij_ref[0,0]", zij_ref[0, 0], tok, [0, 1])}
    out.setdefault("diffusion_conditioning", {})[tag] = rec
    del B, si_ref, zij_ref

json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
