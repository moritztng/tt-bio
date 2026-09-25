#!/usr/bin/env python3
"""of3t-cotcoh steps 1 and 2: decompose the cotangent error into its coherent and isotropic
parts at every block, and read the walk-back curve the pre-registration grades.

Every quantity here is defined in `perf/of3t_cotcoh/PREREGISTRATION.md` and none was added after
a number existed. The reference is upstream 0.4.3's own float64 stack on the same boundary and
the same cotangent; where a ratio is against something else it says so on the line.

CPU only. The device arithmetic this row prices is `ttnn.sum` with `precise_config()`, which
`of3t-lnreduce` measured to be float64-equivalent (relative error exactly 0.0 on all-ones to
K = 147,456), so evaluating the reduction in float64 here is the operator and not a stand-in.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess

import torch

EPS = 1e-5          # of3t-lnreduce's own convention, the LayerNorm eps the stack is built with
NDRAW = 8


def xhat_of(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(dim=-1, keepdim=True)
    c = x - m
    v = (c * c).mean(dim=-1, keepdim=True)
    return c * torch.rsqrt(v + EPS)


def nrm(t) -> float:
    return float(torch.linalg.vector_norm(t))


def spec(E: torch.Tensor):
    s = torch.linalg.svdvals(E)
    s2 = (s * s)
    tot = float(s2.sum())
    if tot <= 0.0:
        return 0.0, 0.0
    return float(s2[0]) / tot, tot / float(s2[0])


def spearman(x, y) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def site(Gd, Gr, Xd, gen):
    """Every pre-registered quantity at one site. Gd, Gr, Xd are [P, C] float64."""
    P, C = Gd.shape
    Xh = xhat_of(Xd)
    E = Gd - Gr

    def R(M):
        return (M * Xh).sum(dim=0)

    nE, nGr, nGd = nrm(E), nrm(Gr), nrm(Gd)
    RE, RGr = R(E), R(Gr)
    nRE, nRGr = nrm(RE), nrm(RGr)

    # --- COH_SPEC and its row-sign-flip control -----------------------------------------
    cs, er = spec(E)
    cs_f, red_f = [], []
    for _ in range(NDRAW):
        s = (torch.randint(0, 2, (P, 1), generator=gen, dtype=torch.int64) * 2 - 1).double()
        Ef = E * s
        a, _b = spec(Ef)
        cs_f.append(a)
        red_f.append(nrm(R(Ef)))
    red_f_sorted = sorted(red_f)
    med_flip = red_f_sorted[len(red_f_sorted) // 2]

    # --- the isotropic amplification, matched in Frobenius norm to E ---------------------
    amp_iso = []
    for _ in range(NDRAW):
        N = torch.randn(P, C, generator=gen, dtype=torch.float64)
        N = N * (nE / nrm(N))
        amp_iso.append((nrm(R(N)) / nE) / (nRGr / nGr) if nGr > 0 and nE > 0 else float("nan"))
    amp_iso_sorted = sorted(amp_iso)
    amp_iso_med = amp_iso_sorted[len(amp_iso_sorted) // 2]
    amp_act = (nRE / nE) / (nRGr / nGr) if nE > 0 and nGr > 0 and nRGr > 0 else float("nan")

    # --- the structure fits ---------------------------------------------------------------
    def expl(fit):
        return 1.0 - (nrm(E - fit) ** 2) / (nE ** 2) if nE > 0 else float("nan")

    gc2 = (Gr * Gr).sum(dim=0)
    d_c = torch.where(gc2 > 0, (E * Gr).sum(dim=0) / gc2.clamp_min(1e-300),
                      torch.zeros_like(gc2))
    S2 = expl(Gr * d_c)
    gt2 = (Gr * Gr).sum(dim=1, keepdim=True)
    d_t = torch.where(gt2 > 0, (E * Gr).sum(dim=1, keepdim=True) / gt2.clamp_min(1e-300),
                      torch.zeros_like(gt2))
    S3 = expl(Gr * d_t)
    a_t = E.mean(dim=1, keepdim=True)
    S4 = expl(a_t.expand_as(E))
    xt2 = (Xh * Xh).sum(dim=1, keepdim=True)
    b_t = (E * Xh).sum(dim=1, keepdim=True) / xt2.clamp_min(1e-300)
    S5 = expl(b_t * Xh)
    alpha = float((E * Gr).sum()) / (nGr ** 2) if nGr > 0 else float("nan")
    S0 = expl(alpha * Gr)

    Epar = alpha * Gr
    Eperp = E - Epar
    return {
        "P": P, "C": C,
        "ref_norm": nGr, "dev_norm": nGd, "err_norm": nE,
        "rel_g_vs_f64": nE / nGr if nGr > 0 else None,
        "cos_g": float((Gd * Gr).sum()) / (nGd * nGr) if nGd > 0 and nGr > 0 else None,
        "R_ref_norm": nRGr, "R_err_norm": nRE,
        "rel_dW_vs_f64_on_device_xhat": nRE / nRGr if nRGr > 0 else None,
        "COH_SPEC": cs, "eff_rank": er,
        "COH_SPEC_flip_median": sorted(cs_f)[len(cs_f) // 2],
        "COH_SPEC_flip_min": min(cs_f), "COH_SPEC_flip_max": max(cs_f),
        "R_flip_median": med_flip, "R_flip_min": red_f_sorted[0], "R_flip_max": red_f_sorted[-1],
        "COH_RED": nRE / med_flip if med_flip > 0 else None,
        "COH_RED_ceiling_sqrtP": math.sqrt(P),
        "AMP_ACT": amp_act, "AMP_ISO_median": amp_iso_med,
        "AMP_ISO_min": amp_iso_sorted[0], "AMP_ISO_max": amp_iso_sorted[-1],
        "AMP_ACT_over_ISO": amp_act / amp_iso_med if amp_iso_med else None,
        "S0_along_ref_global_scale": S0, "S1_rank1": cs, "S2_per_channel_scale": S2,
        "S3_per_position_scale": S3, "S4_position_broadcast": S4, "S5_xhat_aligned": S5,
        "across_share_of_err": (nrm(Eperp) ** 2) / (nE ** 2) if nE > 0 else None,
        "R_from_par": nrm((Epar * Xh).sum(dim=0)), "R_from_perp": nrm((Eperp * Xh).sum(dim=0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260922)
    a = ap.parse_args()

    D = torch.load(a.dev, map_location="cpu", weights_only=False)
    Rf = torch.load(a.ref, map_location="cpu", weights_only=False)
    ds, rs = D["sites"], Rf["sites"]
    gen = torch.Generator().manual_seed(a.seed)

    rows, missing = {}, []
    for fam in ("A", "B"):
        for b in range(48):
            k = f"{fam}:{b}"
            if k not in ds or k not in rs:
                missing.append(k)
                continue
            Gd = ds[k]["g"].to(torch.float64)
            Xd = ds[k]["x"].to(torch.float64)
            Gr = rs[k]["g"].to(torch.float64)
            if Gd.shape != Gr.shape:
                missing.append(k + f" shape {list(Gd.shape)} vs {list(Gr.shape)}")
                continue
            rows[k] = site(Gd, Gr, Xd, gen)
            rows[k]["gamma_path_device"] = ds[k]["gamma_path"]
            rows[k]["outside_mask_sqnorm_device"] = ds[k]["outside_mask_sqnorm"]

    fams = {}
    for fam in ("A", "B"):
        ks = [f"{fam}:{b}" for b in range(48) if f"{fam}:{b}" in rows]
        if not ks:
            continue
        blocks = [int(k.split(":")[1]) for k in ks]
        coh = [rows[k]["COH_RED"] for k in ks]
        depth = [47 - b for b in blocks]
        r = spearman(depth, coh)
        c47 = rows[f"{fam}:47"]["COH_RED"] if f"{fam}:47" in rows else None
        c00 = rows[f"{fam}:0"]["COH_RED"] if f"{fam}:0" in rows else None
        cmax = max(coh)
        if c47 is not None and c47 >= 0.5 * cmax and r < 0.5:
            verdict = "INJECTED"
        elif r >= 0.8 and c47 is not None and c00 is not None and c47 <= 0.25 * c00:
            verdict = "ACCUMULATED"
        else:
            verdict = "MIXED"
        se = sum(rows[k]["err_norm"] ** 2 for k in ks)
        sr = sum(rows[k]["ref_norm"] ** 2 for k in ks)
        sRE = sum(rows[k]["R_err_norm"] ** 2 for k in ks)
        sRr = sum(rows[k]["R_ref_norm"] ** 2 for k in ks)
        sfl = sum(rows[k]["R_flip_median"] ** 2 for k in ks)
        w = [rows[k]["err_norm"] ** 2 for k in ks]
        tw = sum(w) or 1.0
        fams[fam] = {
            "n_sites": len(ks),
            "pooled_rel_g_vs_f64": math.sqrt(se / sr) if sr > 0 else None,
            "pooled_rel_dW_vs_f64_on_device_xhat": math.sqrt(sRE / sRr) if sRr > 0 else None,
            "pooled_COH_RED": math.sqrt(sRE / sfl) if sfl > 0 else None,
            "errmass_weighted": {q: sum(rows[k][q] * wi for k, wi in zip(ks, w)) / tw
                                 for q in ("COH_SPEC", "COH_SPEC_flip_median", "COH_RED",
                                           "AMP_ACT", "AMP_ISO_median", "AMP_ACT_over_ISO",
                                           "S0_along_ref_global_scale", "S2_per_channel_scale",
                                           "S3_per_position_scale", "S4_position_broadcast",
                                           "S5_xhat_aligned", "across_share_of_err",
                                           "eff_rank")},
            "COH_RED_at_47": c47, "COH_RED_at_00": c00, "COH_RED_max": cmax,
            "COH_RED_argmax_block": blocks[coh.index(cmax)],
            "COH_RED_ceiling_sqrtP": rows[ks[0]]["COH_RED_ceiling_sqrtP"],
            "spearman_COH_RED_vs_depth": r,
            "PREREGISTERED_VERDICT": verdict,
            "curve": [{"block": b, "COH_RED": rows[f"{fam}:{b}"]["COH_RED"],
                       "COH_SPEC": rows[f"{fam}:{b}"]["COH_SPEC"],
                       "rel_g": rows[f"{fam}:{b}"]["rel_g_vs_f64"],
                       "rel_dW": rows[f"{fam}:{b}"]["rel_dW_vs_f64_on_device_xhat"],
                       "AMP_ACT_over_ISO": rows[f"{fam}:{b}"]["AMP_ACT_over_ISO"],
                       "S5": rows[f"{fam}:{b}"]["S5_xhat_aligned"],
                       "S4": rows[f"{fam}:{b}"]["S4_position_broadcast"],
                       "S2": rows[f"{fam}:{b}"]["S2_per_channel_scale"]}
                      for b in sorted(blocks)],
        }

    out = {"what": "the coherent and isotropic parts of the cotangent error at the two worst "
                   "affine leaf families, at every block, masked to the real rows",
           "host": socket.gethostname(), "device_involved": False,
           "why_no_aiclk": "CPU only, and every reading is an accuracy reading",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "reference": "upstream OpenFold3 0.4.3 float64 on the same boundary and cotangent",
           "eps": EPS, "draws": NDRAW, "seed": a.seed,
           "dev": a.dev, "ref": a.ref,
           "families": {"A": "attn_pair_bias.layer_norm_a (ours pre_norm_s)",
                        "B": "pair_stack.pair_transition.layer_norm (ours transition_z)"},
           "missing": missing,
           "pad_control_outside_mask_sqnorm_max": max(
               (rows[k]["outside_mask_sqnorm_device"] for k in rows), default=None),
           "family": fams, "sites": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    for fam, f in fams.items():
        print(f"{fam}: n={f['n_sites']} pooled COH_RED={f['pooled_COH_RED']:.4f} "
              f"(ceiling {f['COH_RED_ceiling_sqrtP']:.1f})  at47={f['COH_RED_at_47']} "
              f"at00={f['COH_RED_at_00']} max={f['COH_RED_max']:.4f}@blk"
              f"{f['COH_RED_argmax_block']} spearman={f['spearman_COH_RED_vs_depth']:.4f} "
              f"-> {f['PREREGISTERED_VERDICT']}")
        e = f["errmass_weighted"]
        print(f"   COH_SPEC {e['COH_SPEC']:.4f} vs flip {e['COH_SPEC_flip_median']:.6f} | "
              f"AMP_ACT {e['AMP_ACT']:.4f} / ISO {e['AMP_ISO_median']:.4f} = "
              f"{e['AMP_ACT_over_ISO']:.4f} | across {e['across_share_of_err']:.4f}")
        print(f"   S0 {e['S0_along_ref_global_scale']:.4f} S2 {e['S2_per_channel_scale']:.4f} "
              f"S3 {e['S3_per_position_scale']:.4f} S4 {e['S4_position_broadcast']:.4f} "
              f"S5 {e['S5_xhat_aligned']:.4f}")
    print(json.dumps({"out": a.out, "missing": len(missing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
