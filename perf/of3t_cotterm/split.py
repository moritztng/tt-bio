#!/usr/bin/env python3
"""of3t-cotterm: the ACROSS component of the cotangent error, attributed per contributing term.

`of3t-apbleaf` split the cotangent error at `attn_pair_bias.layer_norm_a` into the part ALONG
the reference cotangent (residue 1.2537x over upstream's own bf16) and the part ACROSS it
(residue 2.3877x, 99.72 % of the dW damage). This attributes the ACROSS part.

Two decompositions of `e = g_dev - g_ref`, both defined in `PREDICTION.md` before any capture
existed, with `F = apbmath.apb_terms`, the float64 APB forward+backward:

  TERMS     e = LOCAL + sum_t [ T_t(ours) - T_t(ref) ],  t in {Q, K, V, GATE}.  An IDENTITY:
            `a` reaches the attention through four linears, so the cotangent it receives is a
            sum of four terms and autograd on four separate leaves splits it exactly.
            LOCAL = g_dev - F(a_d, z_d, do_d, W_d) is every rounding the device's own APB
            arithmetic makes, forward and backward -- `L` at the leaf, one level up.

  CHANNELS  one input at a time from the device: A (single track), Z (pair track, via the raw
            additive bias), DO (inherited cotangent), W (weights). F is nonlinear in these, so
            the sum check is a real check and is reported as one.

Everything is masked to the 56 real rows of 384. The projector is
`P(e) = e - <e, g_ref/||g_ref||> g_ref/||g_ref||`, per site, on the masked rows.
Direction is reported beside magnitude everywhere: the reduction below this site is
cancellation-limited, which is why a rel_l2 alone mis-described the object for five passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch                                                              # noqa: E402

import apbmath                                                            # noqa: E402

PRE = "pairformer_stack.blocks."
LEAF = "attn_pair_bias.layer_norm_a."
BAR = 0.5268825372815341            # the in-frame A26 bar (D218); frame named on every line
SHIPPED = 1.0293953377723410
CEILING = 0.6822397912              # of3t-apbleaf CF_perfect: all 96 at upstream's float64
CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"


def nrm(t):
    return float(torch.linalg.vector_norm(t.reshape(-1)))


def across(e, gref):
    """P(e): the component of e ACROSS gref, both flattened over the masked rows."""
    e, g = e.reshape(-1), gref.reshape(-1)
    gn = float(torch.linalg.vector_norm(g))
    if gn == 0.0:
        return e.clone(), 0.0
    u = g / gn
    al = float(torch.dot(e, u))
    return e - al * u, al


def ln_grads(x, g, eps=1e-5):
    """The LayerNorm affine gradients in float64 from the site's own operands.

    dW = sum_t g_t * xhat_t, db = sum_t g_t. `of3t-apbleaf` validated this reduction against
    torch's float64 autograd through `F.layer_norm` at 3.8e-16 / 7.9e-16 / 6.5e-17 and against
    SWEPT float64 central finite differences.
    """
    x = x.to(torch.float64)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xhat = (x - mu) * torch.rsqrt(var + eps)
    g = g.to(torch.float64)
    flat = lambda t: t.reshape(-1, t.shape[-1])
    return (flat(g) * flat(xhat)).sum(0), flat(g).sum(0)


def validate(ref, sd, tree, block, do, mbias, rows):
    """apbmath against upstream's OWN module, then against SWEPT central differences."""
    out = {"block": block, "tree": tree}
    site = ref["sites"][block]
    W = apbmath.weights_from_sd(sd, f"{PRE}{block}.attn_pair_bias.")
    a = site["a"].to(torch.float64)
    bias = site["bias"].to(torch.float64)
    mine = apbmath.apb_terms(a, bias, mbias, W, do)

    if tree and "z_full" in site:
        sys.path.insert(0, tree)
        import openfold3
        if not openfold3.__file__.startswith(tree):
            raise SystemExit(f"wrong tree: {openfold3.__file__}")
        from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
        heads = W["w_lz"].shape[0]
        m = AttentionPairBias(c_q=a.shape[-1], c_k=a.shape[-1], c_v=a.shape[-1],
                              c_s=a.shape[-1], c_z=site["z_full"].shape[-1],
                              c_hidden=a.shape[-1] // heads, no_heads=heads).to(torch.float64)
        sub = {k[len(f"{PRE}{block}.attn_pair_bias."):]: v.to(torch.float64)
               for k, v in sd.items() if k.startswith(f"{PRE}{block}.attn_pair_bias.")}
        miss, unex = m.load_state_dict(sub, strict=True)
        out["load"] = {"missing": list(miss), "unexpected": list(unex)}
        m.eval()
        z = site["z_full"].to(torch.float64)
        mask = torch.zeros(1, a.shape[1], dtype=torch.float64)
        mask[:, :rows] = 1.0
        # their own two ops on their own z, against apbmath.proj_bias on the same z
        with torch.no_grad():
            theirs = m._prep_bias(a=a, z=z, mask=mask)
        out["proj_bias_vs_upstream_rel"] = nrm(theirs[1] - bias) / max(nrm(theirs[1]), 1e-300)
        out["mask_bias_vs_upstream_rel"] = nrm(theirs[0] - mbias) / max(nrm(theirs[0]), 1e-300)
        aa = a.detach().clone().requires_grad_(True)
        o = m.mha(q_x=aa, kv_x=aa, biases=[mbias.expand_as(theirs[1]), bias])
        out["forward_vs_upstream_rel"] = nrm(o.detach() - mine["out"]) / nrm(o.detach())
        gt, = torch.autograd.grad(o, [aa], grad_outputs=do.to(torch.float64))
        out["cotangent_vs_upstream_autograd_rel"] = nrm(gt - mine["g"]) / nrm(gt)

    # central finite differences of <F(a), do> along one random unit direction, h SWEPT
    torch.manual_seed(0)
    u = torch.randn_like(a)
    u = u / nrm(u)
    analytic = float(torch.dot(mine["g"].reshape(-1), u.reshape(-1)))
    sweep = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        lp = apbmath.apb_forward_scalar(a + h * u, bias, mbias, W, do)
        lm = apbmath.apb_forward_scalar(a - h * u, bias, mbias, W, do)
        fd = (lp - lm) / (2 * h)
        sweep.append({"h": h, "fd": fd, "analytic": analytic,
                      "rel": abs(fd - analytic) / max(abs(analytic), 1e-300)})
    out["fd_sweep"] = sweep
    out["fd_best_rel"] = min(s["rel"] for s in sweep)
    out["fd_at_1e-6_rel"] = [s["rel"] for s in sweep if s["h"] == 1e-6][0]
    out["fd_bar"] = 1e-9
    out["fd_band"] = [min(s["rel"] for s in sweep), max(s["rel"] for s in sweep)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-apb", required=True)
    ap.add_argument("--dev-apb", required=True)
    ap.add_argument("--ref-ln", required=True)
    ap.add_argument("--dev-ln", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--tree", default="")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--validate-block", type=int, default=44)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cf-out", default="", help="the counterfactual dW/db, in cf.py's format; "
                    "one file per name in --cf-channel, the name appended before .pt")
    ap.add_argument("--cf-channel", default="Z",
                    help="comma-separated. A CHANNEL (A/Z/DO/W) removes that input's error "
                         "from the device cotangent; a TERM (T_Q/T_K/T_V/T_GATE) removes that "
                         "additive term's. Everything else, LOCAL included, is left exactly as "
                         "the device produced it")
    a = ap.parse_args()

    ref = torch.load(a.ref_apb, map_location="cpu", weights_only=False)
    dev = torch.load(a.dev_apb, map_location="cpu", weights_only=False)
    rln = torch.load(a.ref_ln, map_location="cpu", weights_only=False)["sites"]
    _draw = torch.load(a.dev_ln, map_location="cpu", weights_only=False)["sites"]
    # armln writes a LIST whose block index lives in `gamma_path`, not a dict
    dln = ({int(c["gamma_path"].split(".")[1]): c for c in _draw}
           if isinstance(_draw, list) else _draw)
    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm = b["single_mask"].reshape(-1)
    rows = int((sm > 0).sum())
    mbias = apbmath.mask_bias(sm)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    blocks = ([int(x) for x in a.blocks.split(",") if x.strip()] if a.blocks
              else sorted(set(int(k) for k in ref["sites"]) & set(int(k) for k in dev["sites"])))
    R = {"what": __doc__.strip().splitlines()[0],
         "host": os.uname().nodename, "ref_host": ref.get("host"), "dev_host": dev.get("host"),
         "board_class": "p150a Blackhole", "real_rows": rows, "padded_width": int(sm.numel()),
         "blocks": blocks, "ref_apb": a.ref_apb, "dev_apb": a.dev_apb,
         "projector": "P(e) = e - <e,u>u, u = g_ref/||g_ref||, on the 56 real rows",
         "frame": "ours against upstream's own bf16 autocast, padded 384 (D218)",
         "bar_in_frame": BAR, "shipped_in_frame": SHIPPED, "leaf_ceiling_in_frame": CEILING}

    # ---- the float64 reference, validated before any arm is read -----------------------
    vb = a.validate_block
    R["validation"] = validate(ref, sd, a.tree, vb,
                               ref["sites"][vb]["do"].to(torch.float64), mbias, rows)

    ARMS = ("REF", "ALL", "A", "Z", "DO", "W")
    CFCH = [c.strip() for c in a.cf_channel.split(",") if c.strip()] if a.cf_out else []
    cfs = {c: {} for c in CFCH}
    per = {}
    mass = {k: 0.0 for k in ("ref", "e", "e_perp", "e_par", "local", "local_perp",
                             "refres")}
    chan = {nm: {"perp2": 0.0, "e2": 0.0} for nm in ARMS if nm not in ("REF",)}
    term = {t: {"perp2": 0.0, "e2": 0.0} for t in ("T_Q", "T_K", "T_V", "T_GATE")}
    sumperp = {"num2": 0.0}
    track = {"dbias_e2": 0.0, "dbias_ref2": 0.0, "z_e2": 0.0, "z_ref2": 0.0}

    for i in blocks:
        rs, ds = ref["sites"][i], dev["sites"][i]
        W_r = apbmath.weights_from_sd(sd, f"{PRE}{i}.attn_pair_bias.")
        W_d = {k: (v.to(torch.bfloat16).to(torch.float64) if v.dim() else v)
               for k, v in W_r.items()}
        ins = {"REF": (rs["a"], rs["bias"], rs["do"], W_r),
               "ALL": (ds["a"], ds["bias"], ds["do"], W_d),
               "A":   (ds["a"], rs["bias"], rs["do"], W_r),
               "Z":   (rs["a"], ds["bias"], rs["do"], W_r),
               "DO":  (rs["a"], rs["bias"], ds["do"], W_r),
               "W":   (rs["a"], rs["bias"], rs["do"], W_d)}
        res = {nm: apbmath.apb_terms(v[0].to(torch.float64), v[1].to(torch.float64),
                                     mbias, v[3], v[2].to(torch.float64))
               for nm, v in ins.items()}
        m = slice(0, rows)
        # the reference direction is the cotangent the reference ARM recorded, not a
        # recomputation of it; the recomputation is scored against it as a control
        gref = rln[i]["g"].to(torch.float64)[:, m].reshape(-1)
        grecomp = res["REF"]["g"][:, m].reshape(-1)
        gdev = dln[i]["g"].to(torch.float64)[:, m].reshape(-1)
        e_tot = gdev - gref
        p_tot, al_tot = across(e_tot, gref)
        loc = gdev - res["ALL"]["g"][:, m].reshape(-1)
        p_loc, _ = across(loc, gref)

        row = {"block": i, "ref_g_norm": nrm(gref), "dev_g_norm": nrm(gdev),
               "e_norm": nrm(e_tot), "e_perp_norm": nrm(p_tot), "e_par_norm": abs(al_tot),
               "across_share_pct": 100.0 * nrm(p_tot) ** 2 / max(nrm(e_tot) ** 2, 1e-300),
               "cos_dev_ref": float(torch.dot(gdev, gref)) / max(nrm(gdev) * nrm(gref), 1e-300),
               "rel_l2": nrm(e_tot) / max(nrm(gref), 1e-300),
               "local_norm": nrm(loc), "local_perp_norm": nrm(p_loc),
               "local_share_of_e_pct": 100.0 * nrm(loc) / max(nrm(e_tot), 1e-300),
               "ref_recompute_rel": nrm(grecomp - gref) / max(nrm(gref), 1e-300),
               "channels": {}, "terms": {}}
        mass["ref"] += nrm(gref) ** 2
        mass["e"] += nrm(e_tot) ** 2
        mass["e_perp"] += nrm(p_tot) ** 2
        mass["e_par"] += al_tot ** 2
        mass["local"] += nrm(loc) ** 2
        mass["local_perp"] += nrm(p_loc) ** 2
        mass["refres"] += nrm(grecomp - gref) ** 2

        ssum = torch.zeros_like(p_tot)
        for nm in ARMS:
            if nm == "REF":
                continue
            e_c = res[nm]["g"][:, m].reshape(-1) - grecomp
            p_c, al_c = across(e_c, gref)
            chan[nm]["perp2"] += nrm(p_c) ** 2
            chan[nm]["e2"] += nrm(e_c) ** 2
            row["channels"][nm] = {"e_norm": nrm(e_c), "e_perp_norm": nrm(p_c),
                                   "e_par_norm": abs(al_c),
                                   "cos_with_e_perp_total":
                                       float(torch.dot(p_c, p_tot))
                                       / max(nrm(p_c) * nrm(p_tot), 1e-300)}
            if nm != "ALL":
                ssum = ssum + p_c
        sumperp["num2"] += nrm(ssum) ** 2
        row["channel_sum_vs_ALL"] = nrm(ssum) / max(nrm(res["ALL"]["g"][:, m].reshape(-1)
                                                       - gref), 1e-300)

        for t in term:
            e_t = (res["ALL"][t][:, m].reshape(-1) - res["REF"][t][:, m].reshape(-1))
            p_t, al_t = across(e_t, gref)
            term[t]["perp2"] += nrm(p_t) ** 2
            term[t]["e2"] += nrm(e_t) ** 2
            row["terms"][t] = {"e_norm": nrm(e_t), "e_perp_norm": nrm(p_t),
                              "ref_term_norm": nrm(res["REF"][t][:, m])}

        db_r = res["REF"]["dbias"][:, :, m, m]
        db_d = res["ALL"]["dbias"][:, :, m, m]
        track["dbias_e2"] += nrm(db_d - db_r) ** 2
        track["dbias_ref2"] += nrm(db_r) ** 2
        row["dbias_rel_l2"] = nrm(db_d - db_r) / max(nrm(db_r), 1e-300)
        row["dbias_cos"] = (float(torch.dot(db_d.reshape(-1), db_r.reshape(-1)))
                            / max(nrm(db_d) * nrm(db_r), 1e-300))
        if "z_real" in rs and "z_real" in ds:
            zr = rs["z_real"].to(torch.float64)
            zd = ds["z_real"].to(torch.float64)
            track["z_e2"] += nrm(zd - zr) ** 2
            track["z_ref2"] += nrm(zr) ** 2
            row["z_rel_l2"] = nrm(zd - zr) / max(nrm(zr), 1e-300)
            row["z_cos"] = (float(torch.dot(zd.reshape(-1), zr.reshape(-1)))
                            / max(nrm(zd) * nrm(zr), 1e-300))
        per[i] = row
        # The counterfactual is built HERE, per block, rather than by keeping every arm's
        # tensors to the end: at 48 blocks that is about 4 GB of float64 nobody reads twice.
        for c in CFCH:
            # c's OWN error, the same quantity the shares above are taken of: a channel's is
            # `res[c] - res[REF]`, one input at a time from the device, and a term's is
            # `res[ALL][t] - res[REF][t]`. The inherited `--cf-out` used `res[ALL] - res[c]`,
            # which is every channel EXCEPT c -- the opposite arm, and it ranked Z above A
            # while the shares rank A five times above Z.
            # NULL removes nothing. It is the control: cf.py REPLACES the device's own dW with
            # this file's, so every arm also silently drops the three device ops' rounding at
            # the leaf. NULL carries that and only that, so a channel is read against NULL.
            if c == "NULL":
                corr = torch.zeros_like(res["REF"]["g"])
            elif c.startswith("T_"):
                corr = res["ALL"][c] - res["REF"][c]
            else:
                corr = res[c]["g"] - res["REF"]["g"]
            gdev_full = dln[i]["g"].to(torch.float64)
            g_cf = gdev_full.clone()
            g_cf[:, :rows] = gdev_full[:, :rows] - corr[:, :rows]
            dW, db = ln_grads(dln[i]["x"].to(torch.float64), g_cf)
            cfs[c][f"{PRE}{i}.{LEAF}weight"] = dW
            cfs[c][f"{PRE}{i}.{LEAF}bias"] = db
        print("blk %2d  ||e||=%.6e  ||e_perp||=%.6e (%.2f %%)  LOCAL=%.3e  "
              "Z_perp=%.3e A_perp=%.3e DO_perp=%.3e W_perp=%.3e"
              % (i, row["e_norm"], row["e_perp_norm"], row["across_share_pct"],
                 row["local_norm"], row["channels"]["Z"]["e_perp_norm"],
                 row["channels"]["A"]["e_perp_norm"], row["channels"]["DO"]["e_perp_norm"],
                 row["channels"]["W"]["e_perp_norm"]), flush=True)

    S = lambda x: math.sqrt(max(x, 0.0))
    den = S(chan["ALL"]["perp2"])
    R["pooled"] = {
        "denominator": "the ACROSS mass of the ALL arm, pooled over the %d sites, "
                       "masked to the %d real rows" % (len(blocks), rows),
        "ref_norm": S(mass["ref"]), "e_norm": S(mass["e"]),
        "e_perp_norm": S(mass["e_perp"]), "e_par_norm": S(mass["e_par"]),
        "across_share_of_e_pct": 100.0 * mass["e_perp"] / max(mass["e"], 1e-300),
        "local_norm": S(mass["local"]), "local_perp_norm": S(mass["local_perp"]),
        "local_share_of_e": S(mass["local"]) / max(S(mass["e"]), 1e-300),
        "ref_recompute_rel": S(mass["refres"]) / max(S(mass["ref"]), 1e-300),
        "ALL_perp_norm": den,
        "channel_sum_over_ALL": S(sumperp["num2"]) / max(den, 1e-300)}
    R["channels"] = {nm: {"e_perp_norm": S(v["perp2"]), "e_norm": S(v["e2"]),
                          "share_of_ALL_across": S(v["perp2"]) / max(den, 1e-300)}
                     for nm, v in chan.items()}
    R["terms"] = {t: {"e_perp_norm": S(v["perp2"]), "e_norm": S(v["e2"]),
                      "share_of_ALL_across": S(v["perp2"]) / max(den, 1e-300)}
                  for t, v in term.items()}
    R["track"] = {"dbias_rel_l2": S(track["dbias_e2"]) / max(S(track["dbias_ref2"]), 1e-300),
                  "dbias_abs_err": S(track["dbias_e2"]),
                  "dbias_ref_norm": S(track["dbias_ref2"]),
                  "z_rel_l2": (S(track["z_e2"]) / max(S(track["z_ref2"]), 1e-300)
                               if track["z_ref2"] else None),
                  "z_abs_err": S(track["z_e2"]), "z_ref_norm": S(track["z_ref2"]),
                  "why_not_R44": "R44 is ds at rung 44, a single-track metric. D231 measured "
                                 "ds at exactly 0.0 on all 49 rungs for both pins, and "
                                 "of3t-readverbs doubled the cotangent at 960 sites leaving all "
                                 "49 ds rungs bit-identical while 48 of 49 dz rungs diverged. "
                                 "dbias is what this backward EXPORTS to the pair track, which "
                                 "is the dz reading at this site."}
    worst = max(per.values(), key=lambda r: r["e_perp_norm"])
    R["worst_case"] = {"block": worst["block"],
                       "located": f"{PRE}{worst['block']}.{LEAF}weight",
                       "e_perp_norm": worst["e_perp_norm"], "rel_l2": worst["rel_l2"],
                       "cos_dev_ref": worst["cos_dev_ref"],
                       "channels": worst["channels"], "terms": worst["terms"]}
    ch = {nm: v["share_of_ALL_across"] for nm, v in R["channels"].items() if nm != "ALL"}
    tm = {t: v["share_of_ALL_across"] for t, v in R["terms"].items()}
    top_c = max(ch, key=ch.get)
    top_t = max(tm, key=tm.get)
    R["verdict_inputs"] = {
        "channel_top": top_c, "channel_top_share": ch[top_c],
        "term_top": top_t, "term_top_share": tm[top_t],
        "named_at": 0.60, "spread_at": 0.40,
        "channel_call": ("NAMED:" + top_c if ch[top_c] >= 0.60
                         else ("SPREAD" if max(ch.values()) <= 0.40 else "SHARE")),
        "term_call": ("NAMED:" + top_t if tm[top_t] >= 0.60
                      else ("SPREAD" if max(tm.values()) <= 0.40 else "SHARE")),
        "secondary_1_local_under_0.25": R["pooled"]["local_share_of_e"] <= 0.25,
        "secondary_2_gate_under_0.25": tm["T_GATE"] <= 0.25,
        "secondary_3_sum_check_in_band":
            0.5 <= R["pooled"]["channel_sum_over_ALL"] <= 2.0}
    R["per_site"] = per

    if a.cf_out:
        base = a.cf_out[:-3] if a.cf_out.endswith(".pt") else a.cf_out
        R["cf_out"] = []
        for c, cf in cfs.items():
            path = f"{base}_{c}.pt"
            torch.save({"dW_f64dev": cf, "channel": c, "host": os.uname().nodename,
                        "blocks": blocks, "real_rows": rows,
                        "what": "device cotangent with %s's error removed" % c}, path)
            R["cf_out"].append({"path": path, "channel": c, "tensors": len(cf)})

    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps({k: R[k] for k in ("pooled", "channels", "terms", "track",
                                        "verdict_inputs")}, indent=2, default=str)[:4000])
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
