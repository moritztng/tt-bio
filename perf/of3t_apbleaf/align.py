#!/usr/bin/env python3
"""of3t-apbleaf: why our dW residue is 2.4044x upstream's when our cotangent is only 1.1745x.

`dW = sum_t g_t xhat_t` is LINEAR in `g`, so with `xhat` held at the float64 reference's own the
error in `dW` caused by an error `e = g - g_ref` is exactly `R(e)`, and `R` splits with `e`:

    e_par   = (e . ghat_ref) ghat_ref      the part along the reference cotangent
    e_perp  = e - e_par                    the part across it
    R(e)    = R(e_par) + R(e_perp)         exactly, no approximation

A cotangent error ALONG the reference is a scale error and passes through the reduction at its
own size. One ACROSS it is a new direction and is multiplied by the reduction's cancellation.
Two cotangents with the same rel_l2 and different cosines therefore land different dW errors,
which is the arithmetic that lets a 1.1745x cotangent residue become a 2.4044x gradient residue.

Same operator for both sides, so the only thing that differs is `g`.
"""
import argparse, json, math
import torch


def xhat_of(x, eps):
    x = x.to(torch.float64)
    mu = x.mean(-1, keepdim=True); xc = x - mu
    return xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)


def R(e, xhat):
    c = e.shape[-1]
    return (e.to(torch.float64) * xhat).reshape(-1, c).sum(0)


def split(g, gref, xhat):
    e = g.to(torch.float64).reshape(-1) - gref.to(torch.float64).reshape(-1)
    gh = gref.to(torch.float64).reshape(-1)
    gh = gh / (gh.norm() + 1e-300)
    par = (e @ gh) * gh
    perp = e - par
    sh = gref.shape
    return {"e_norm": float(e.norm()), "e_par_norm": float(par.norm()),
            "e_perp_norm": float(perp.norm()),
            "R_e": float(R(e.reshape(sh), xhat).norm()),
            "R_e_par": float(R(par.reshape(sh), xhat).norm()),
            "R_e_perp": float(R(perp.reshape(sh), xhat).norm())}


ap = argparse.ArgumentParser()
ap.add_argument("--dev-ln", required=True)
ap.add_argument("--ref-ln", required=True)
ap.add_argument("--bf16-ln", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.load(a.dev_ln, map_location="cpu", weights_only=False)
ref = torch.load(a.ref_ln, map_location="cpu", weights_only=False)["sites"]
bf = torch.load(a.bf16_ln, map_location="cpu", weights_only=False)["sites"]
ds = {int(s["gamma_path"].split(".")[1]): s for s in dev["sites"]}
rs = {int(k): v for k, v in ref.items()}
bs = {int(k): v for k, v in bf.items()}

R_ = {"what": __doc__.strip().splitlines()[0], "by_block": {}}
acc = {k: 0.0 for k in ("o_par", "o_perp", "o_all", "u_par", "u_perp", "u_all", "refdW")}
for b in sorted(set(ds) & set(rs) & set(bs)):
    xh = xhat_of(rs[b]["x"], 1e-5)
    o = split(ds[b]["g"], rs[b]["g"], xh)
    u = split(bs[b]["g"], rs[b]["g"], xh)
    rn = float(rs[b]["dW_ref"].norm())
    R_["by_block"][b] = {"ours": o, "upstream_bf16": u, "ref_dW_norm": rn,
                         "ours_perp_share": o["R_e_perp"] / (o["R_e"] + 1e-300),
                         "upstream_perp_share": u["R_e_perp"] / (u["R_e"] + 1e-300)}
    acc["o_par"] += o["R_e_par"] ** 2; acc["o_perp"] += o["R_e_perp"] ** 2
    acc["o_all"] += o["R_e"] ** 2
    acc["u_par"] += u["R_e_par"] ** 2; acc["u_perp"] += u["R_e_perp"] ** 2
    acc["u_all"] += u["R_e"] ** 2
    acc["refdW"] += rn ** 2
s = lambda k: math.sqrt(acc[k])
R_["mass_weighted"] = {
    "ours_dW_error_from_g": s("o_all"), "ours_from_e_par": s("o_par"),
    "ours_from_e_perp": s("o_perp"),
    "upstream_dW_error_from_g": s("u_all"), "upstream_from_e_par": s("u_par"),
    "upstream_from_e_perp": s("u_perp"),
    "residue_total": s("o_all") / s("u_all"),
    "residue_in_the_across_part": s("o_perp") / s("u_perp"),
    "residue_in_the_along_part": s("o_par") / s("u_par"),
    "ours_across_share": s("o_perp") / s("o_all"),
    "upstream_across_share": s("u_perp") / s("u_all"),
    "ref_dW_norm": s("refdW"),
    "denominator": "the 48 captured sites, squared sums, one reduction operator for both sides"}
print(json.dumps(R_["mass_weighted"], indent=1))
print("\n blk | ours R(e) R(par) R(perp) across%% | upstream R(e) R(par) R(perp) across%%")
for b in (44, 4, 0, 42, 38, 15):
    r = R_["by_block"][b]
    print(" %3d | %.3e %.3e %.3e %6.2f%% | %.3e %.3e %.3e %6.2f%%"
          % (b, r["ours"]["R_e"], r["ours"]["R_e_par"], r["ours"]["R_e_perp"],
             100 * r["ours_perp_share"], r["upstream_bf16"]["R_e"],
             r["upstream_bf16"]["R_e_par"], r["upstream_bf16"]["R_e_perp"],
             100 * r["upstream_perp_share"]))
json.dump(R_, open(a.out, "w"), indent=2)
print("wrote " + a.out)
