"""Score the pre-registered revision arm. Control first; branches are read only if it passes."""
import json, torch

L = lambda p: torch.load(p, map_location="cpu", weights_only=False)
D = "/tmp/of3t/revarm/"
a43, a50, a50off, abf = (L(D+f) for f in ("out_043_fp32.pt","out_050_fp32.pt","out_050_tboff.pt","out_043_bf16.pt"))

def dist(e, t):
    e, t = e.double(), t.double()
    d = e - t
    ne, nt, nd = float(e.norm()), float(t.norm()), float(d.norm())
    cos = float((d.flatten() @ t.flatten()) / (nd * nt)) if nd > 0 else 0.0
    return {"rel_vs_t": nd/nt, "rel_vs_e": nd/ne, "norm_ratio_e_over_t": ne/nt,
            "cos_err_vs_t": cos, "abs_l2": nd}

out = {"as_of": "2026-09-20 pass 176", "boundary": "real 0.4.3 trunk entry, 5nw3/56 tok padded 384",
       "weights": "of3-p2-155k.pt, identical tensors into both stacks, 2736 loaded, strict=True"}

out["CONTROL_050_tboff_vs_043"] = {k: dist(a50off[k], a43[k]) for k in ("s","z")}
ctl_z = out["CONTROL_050_tboff_vs_043"]["z"]["abs_l2"]
ctl_s = out["CONTROL_050_tboff_vs_043"]["s"]["abs_l2"]
out["control_exactly_zero"] = (ctl_z == 0.0 and ctl_s == 0.0)

out["REVISION_050_vs_043_fp32"] = {k: dist(a50[k], a43[k]) for k in ("s","z")}
out["UPSTREAM_OWN_BF16"] = {k: dist(abf[k], a43[k]) for k in ("s","z")}

for k in ("s","z"):
    r = out["REVISION_050_vs_043_fp32"][k]["rel_vs_t"]
    b = out["UPSTREAM_OWN_BF16"][k]["rel_vs_t"]
    out.setdefault("R_in_bf16_units", {})[k] = r/b if b > 0 else None
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/of3t/revarm/score.json","w"), indent=1)
