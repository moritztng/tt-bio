#!/usr/bin/env python3
"""Every op of the AttentionPairBias backward, scored against float64 on captured operands.

Two scorings, both defined in `PREREGISTERED.md` before any number existed.

ISOLATION. For each taped op the device ran, recompute THAT op's backward in float64 from the
operands the device actually had and the cotangent the device actually handed it, and compare
against what the device wrote. This is `of3t-bwdaccum`'s LayerNorm-leaf method applied op by
op, and it answers one question per op: is this op's arithmetic wrong.

COMPOSED WALK. Recompute the WHOLE module's backward in float64 from the device's captured
forward operands and the device's incoming cotangent at the module output, and compare at every
node. Walking it backwards names the first place device and float64 part company. Where the two
scorings disagree the walk is the stronger one: an op can be individually blameless on a wrong
input and the walk is what carries that in.

The scalar multiply is scored against the scalar the CARD APPLIED, read out of the captured
forward (out/in) rather than out of the source. `ttnn.multiply_` truncates a python float's
mantissa to bf16 on a bf16 tensor, so the implemented function is not the written one, and a
backward that matches the written one would be the wrong answer.
"""
from __future__ import annotations

import argparse
import json
import math

import torch


def stats(dev, ref):
    """rel_l2, norm ratio and cosine. D35: never rel_l2 alone."""
    if dev is None or ref is None:
        return None
    d, r = dev.reshape(-1).to(torch.float64), ref.reshape(-1).to(torch.float64)
    nr, nd = float(r.norm()), float(d.norm())
    den = nr if nr > 0 else 1.0
    cos = float((d @ r) / (nd * nr)) if nd > 0 and nr > 0 else float("nan")
    return {"rel": float((d - r).norm() / den), "r": (nd / nr) if nr > 0 else float("inf"),
            "cos": cos, "ref_norm": nr, "dev_norm": nd}


# ---- float64 op rules, each the exact backward of the op the device ran -----------------

def bw_linear(ins, g):
    x, w = ins[0], ins[1]
    b = ins[2] if len(ins) > 2 else None
    lead = x.shape[:-1]
    xf, gf = x.reshape(-1, x.shape[-1]), g.reshape(-1, g.shape[-1])
    out = [g @ w.transpose(-1, -2), xf.transpose(0, 1) @ gf]
    if b is not None:
        out.append(gf.sum(0).reshape(b.shape))
    return out


def bw_reshape(ins, g):
    return [g.reshape(ins[0].shape)]


def bw_permute(ins, g, dims):
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    return [g.permute(*inv).contiguous()]


def bw_transpose(ins, g, d0, d1):
    return [g.transpose(d0, d1).contiguous()]


def bw_matmul(ins, g):
    a, b = ins[0], ins[1]
    return [g @ b.transpose(-1, -2), a.transpose(-1, -2) @ g]


def reduce_to(g, shape):
    while g.dim() > len(shape):
        g = g.sum(0)
    for i, n in enumerate(shape):
        if g.shape[i] != n:
            g = g.sum(i, keepdim=True)
    return g.reshape(shape)


def bw_add(ins, g):
    return [g] + [reduce_to(g, t.shape) for t in ins[1:]]


def bw_softmax(y, g, dim=-1):
    return [y * (g - (g * y).sum(dim, keepdim=True))]


def bw_layer_norm(ins, g, eps=1e-5):
    x, gamma = ins[0], ins[1]
    beta = ins[2] if len(ins) > 2 else None
    mean = x.mean(-1, keepdim=True)
    cen = x - mean
    var = (cen * cen).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = cen * rstd
    dn = g * gamma if gamma is not None else g
    dx = (dn - dn.mean(-1, keepdim=True) - xhat * (dn * xhat).mean(-1, keepdim=True)) * rstd
    flat_g = g.reshape(-1, g.shape[-1])
    out = [dx, (flat_g * xhat.reshape(-1, x.shape[-1])).sum(0).reshape(gamma.shape)]
    if beta is not None:
        out.append(flat_g.sum(0).reshape(beta.shape))
    return out


def bw_gate(ins, g):
    """`multiply(o, gpre, input_tensor_b_activations=[SIGMOID])`: out = o * sigmoid(gpre)."""
    o, gp = ins[0], ins[1]
    s = torch.sigmoid(gp)
    return [g * s, g * o * s * (1.0 - s)]


def bw_qkv_heads(x, gs, H):
    """`nlp_create_qkv_heads`: [B,1,L,3HD] -> 3x[B,H,L,D], slot-major over the packed axis."""
    B, _, L, wide = x.shape
    D = wide // (3 * H)
    out = torch.zeros(B, L, 3, H, D, dtype=torch.float64)
    for s, g in enumerate(gs):
        if g is not None:
            out[:, :, s] = g.permute(0, 2, 1, 3)
    return [out.reshape(B, 1, L, 3 * H * D)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", required=True)
    ap.add_argument("--refsites", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--break-permute", type=int, default=0, metavar="SEED",
                    help="BREAK CONTROL: permute the captured cotangent's real token "
                         "positions before the float64 recomputation. The isolation reading "
                         "must move, or it reads nothing.")
    ap.add_argument("--real-tokens", type=int, default=56)
    a = ap.parse_args()

    d = torch.load(a.tap, map_location="cpu", weights_only=False)
    recs, meta = d["recs"], d["meta"]
    refsite = {}
    if a.refsites:
        rs = torch.load(a.refsites, map_location="cpu", weights_only=False)
        refsite = rs.get("site_cot", {})

    perm = None
    if a.break_permute:
        gg = torch.Generator().manual_seed(a.break_permute)
        real = torch.arange(a.real_tokens)
        perm = torch.arange(64)
        perm[:a.real_tokens] = real[torch.randperm(a.real_tokens, generator=gg)]

    def scramble(t):
        """Permute the TOKEN axes of a cotangent. Shapes are known and few."""
        if perm is None or t is None:
            return t
        s = list(t.shape)
        if len(s) == 3 and s[1] == 64:
            return t[:, perm].contiguous()
        if len(s) == 4 and s[2] == 64 and s[3] == 64:
            return t[:, :, perm][:, :, :, perm].contiguous()
        if len(s) == 4 and s[1] == 64 and s[2] == 64:
            return t[:, perm][:, :, perm].contiguous()
        if len(s) == 4 and s[2] == 64:
            return t[:, :, perm].contiguous()
        return t

    report = {"what": __doc__.strip().splitlines()[0], "tap": a.tap,
              "break_permute": a.break_permute, "blocks": {}}

    for blk in sorted(recs):
        M = meta[blk]
        H, hd, D = M["n_heads"], M["head_dim"], M["padded_head_dim"]
        # the pass whose backward actually fired (the z-seeded recompute builds the tape and
        # never traverses it, because the block's z output does not depend on this module)
        passes = [p for p in recs[blk] if any(o["g"] for r in p["ops"] for o in r["out"])]
        if not passes:
            continue
        P = passes[0]
        ops = P["ops"]
        by_idx = {r["idx"]: r for r in ops}

        def G(idx, j=0):
            gs = by_idx[idx]["out"][j]["g"]
            return None if not gs else sum(gs[1:], gs[0])

        def C(idx, j=0):
            """device contributions of op idx output j, as {operand_id: tensor}."""
            out = {}
            for oid, t in by_idx[idx]["out"][j]["contrib"]:
                out[oid] = t if oid not in out else out[oid] + t
            return out

        def IN(idx):
            return [i["val"] for i in by_idx[idx]["in"]]

        def IID(idx):
            return [i["id"] for i in by_idx[idx]["in"]]

        def OUT(idx, j=0):
            return by_idx[idx]["out"][j]["val"]

        # ---------- ISOLATION -------------------------------------------------------
        iso = []

        def score(idx, name, refs, j=0, names=None):
            """`refs` is the float64 backward's contributions, in operand order."""
            dev = C(idx, j)
            ids = IID(idx)
            for k, rf in enumerate(refs):
                if rf is None or k >= len(ids) or ids[k] is None:
                    continue
                dv = dev.get(ids[k])
                if dv is None:
                    continue
                st = stats(dv.reshape(rf.shape), rf)
                iso.append({"op": name, "idx": idx, "operand": (names[k] if names else str(k)),
                            "shape": list(rf.shape), **st})

        gx = scramble(G(28))
        score(28, "linear:o_proj", bw_linear(IN(28), gx), names=["og", "o_weight"])
        g26 = scramble(G(26))
        score(26, "multiply:gate(sigmoid)", bw_gate(IN(26), g26), names=["o", "g_pre"])
        g25 = scramble(G(25))
        score(25, "linear:g_proj", bw_linear(IN(25), g25), names=["s_norm", "g_weight"])
        g24 = scramble(G(24))
        score(24, "permute:heads_out", bw_permute(IN(24), g24, [0, 2, 1]), names=["x"])
        g23 = scramble(G(23))
        score(23, "reshape:heads_out", bw_reshape(IN(23), g23), names=["x"])
        g22 = scramble(G(22))
        score(22, "permute:heads_dh", bw_permute(IN(22), g22, [0, 1, 3, 2]), names=["x"])
        g17 = scramble(G(17))
        score(17, "matmul:probs@v", bw_matmul(IN(17), g17), names=["probs", "v"])
        g15 = scramble(G(15))
        score(15, "softmax", bw_softmax(OUT(15), g15), names=["logits"])
        g14 = scramble(G(14))
        # the scalar the CARD applied, read out of the forward it performed
        xin, xout = IN(14)[0], OUT(14)
        m = (xin.abs() > 0)
        f_fwd = float((xout[m] / xin[m]).median()) if m.any() else float("nan")
        score(14, "multiply_:score_scale", [g14 * f_fwd], names=["logits"])
        g13 = scramble(G(13))
        score(13, "add_:logits+bias", bw_add(IN(13), g13), names=["qk", "bias"])
        g11 = scramble(G(11))
        score(11, "matmul:q@kT", bw_matmul(IN(11), g11), names=["q", "kT"])
        g10 = scramble(G(10))
        score(10, "transpose:k", bw_transpose(IN(10), g10, -2, -1), names=["k"])
        g9 = scramble(G(9))
        score(9, "add_:bias+seq_mask", bw_add(IN(9), g9), names=["bias0", "seq_mask"])
        g7 = scramble(G(7))
        score(7, "permute:bias", bw_permute(IN(7), g7, [0, 3, 1, 2]), names=["zb"])
        g5 = scramble(G(5))
        score(5, "linear:z_proj", bw_linear(IN(5), g5), names=["z_norm", "z_weight"])
        g4 = scramble(G(4))
        score(4, "layer_norm:z", bw_layer_norm(IN(4), g4),
              names=["z", "z_norm_weight", "z_norm_bias"])
        gqkv = [scramble(G(2, j)) for j in range(3)]
        for j in range(3):
            if gqkv[j] is None:
                continue
            ref = bw_qkv_heads(IN(2)[0], [gqkv[j] if i == j else None for i in range(3)], H)
            score(2, f"nlp_create_qkv_heads:slot{j}", ref, j=j, names=["qkv4"])
        g1 = scramble(G(1))
        score(1, "unsqueeze:qkv", bw_reshape(IN(1), g1), names=["qkv"])
        g0 = scramble(G(0))
        score(0, "linear:qkv_proj", bw_linear(IN(0), g0),
              names=["s_norm", "qkv_weight", "qkv_bias"])

        # ---------- COMPOSED WALK ---------------------------------------------------
        # float64 backward of the WHOLE module from the device's forward operands and the
        # device's incoming cotangent at the module output.
        s_norm = P["s_in"]
        Wqkv, bqkv = IN(0)[1], IN(0)[2]
        Wz, Wg, Wo = IN(5)[1], IN(25)[1], IN(28)[1]
        gz, bz = IN(4)[1], IN(4)[2]
        z = IN(4)[0]
        seq_mask = IN(9)[1]
        dx = G(28)

        qkv = s_norm @ Wqkv + bqkv
        B, L, wide = qkv.shape
        pk = qkv.reshape(B, L, 3, H, D)
        q = pk[:, :, 0].permute(0, 2, 1, 3).contiguous()
        k = pk[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v = pk[:, :, 2].permute(0, 2, 1, 3).contiguous()
        mu = z.mean(-1, keepdim=True)
        cen = z - mu
        rstd = torch.rsqrt((cen * cen).mean(-1, keepdim=True) + 1e-5)
        zhat = cen * rstd
        z_n = zhat * gz + bz
        zb = z_n @ Wz
        bias = zb.permute(0, 3, 1, 2) + seq_mask
        qk = q @ k.transpose(-1, -2)
        logits = (qk + bias) * f_fwd
        probs = torch.softmax(logits, dim=-1)
        o32 = probs @ v
        o24 = o32[..., :hd]
        o = o24.permute(0, 1, 3, 2).reshape(B, H * hd, L).permute(0, 2, 1).contiguous()
        gpre = s_norm @ Wg
        sig = torch.sigmoid(gpre)
        og = o * sig

        walk = {}
        d_og = dx @ Wo.transpose(-1, -2)
        walk["og"] = d_og
        walk["o"] = d_og * sig
        walk["g_pre"] = d_og * o * sig * (1.0 - sig)
        ds_g = walk["g_pre"] @ Wg.transpose(-1, -2)
        d_o24 = walk["o"].permute(0, 2, 1).reshape(B, H, hd, L).permute(0, 1, 3, 2).contiguous()
        d_o32 = torch.zeros_like(o32)
        d_o32[..., :hd] = d_o24
        walk["o32"] = d_o32
        d_probs = d_o32 @ v.transpose(-1, -2)
        d_v = probs.transpose(-1, -2) @ d_o32
        walk["probs"] = d_probs
        d_logits = probs * (d_probs - (d_probs * probs).sum(-1, keepdim=True))
        walk["logits"] = d_logits
        d_pre = d_logits * f_fwd
        walk["qk"] = d_pre
        walk["bias"] = d_pre
        d_q = d_pre @ k
        d_k = d_pre.transpose(-1, -2) @ q
        walk["q"], walk["k"], walk["v"] = d_q, d_k, d_v
        d_pack = torch.zeros(B, L, 3, H, D, dtype=torch.float64)
        d_pack[:, :, 0] = d_q.permute(0, 2, 1, 3)
        d_pack[:, :, 1] = d_k.permute(0, 2, 1, 3)
        d_pack[:, :, 2] = d_v.permute(0, 2, 1, 3)
        d_qkv = d_pack.reshape(B, L, 3 * H * D)
        walk["qkv"] = d_qkv
        ds_qkv = d_qkv @ Wqkv.transpose(-1, -2)
        walk["s_norm"] = ds_qkv + ds_g
        walk["ds_qkv"] = ds_qkv
        walk["ds_g"] = ds_g

        # the device's own value at each of those nodes
        dev_node = {
            "og": C(28).get(IID(28)[0]),
            "o": C(26).get(IID(26)[0]),
            "g_pre": C(26).get(IID(26)[1]),
            "probs": C(17).get(IID(17)[0]),
            "v": C(17).get(IID(17)[1]),
            "logits": C(15).get(IID(15)[0]),
            "qk": C(13).get(IID(13)[0]),
            "bias": C(13).get(IID(13)[1]),
            "q": C(11).get(IID(11)[0]),
            "qkv": C(1).get(IID(1)[0]),
        }
        dev_node["k"] = C(10).get(IID(10)[0])
        s_id = P["s_in_id"]
        dev_ds_qkv = C(0).get(s_id)
        dev_ds_g = C(25).get(s_id)
        dev_node["ds_qkv"] = dev_ds_qkv
        dev_node["ds_g"] = dev_ds_g
        dev_node["s_norm"] = (None if dev_ds_qkv is None or dev_ds_g is None
                              else dev_ds_qkv + dev_ds_g)

        walk_rows = []
        for nm in ("og", "o", "g_pre", "probs", "v", "logits", "qk", "bias", "q", "k",
                   "qkv", "ds_qkv", "ds_g", "s_norm"):
            dv = dev_node.get(nm)
            if dv is None:
                continue
            walk_rows.append({"node": nm, **stats(dv.reshape(walk[nm].shape), walk[nm])})

        # the REFERENCE's own cotangent at the module's LayerNorm-ed input
        ref_key = f"blocks.{blk}.pre_norm_s"
        ref_ds = refsite.get(ref_key)
        vs_ref = {}
        if ref_ds is not None:
            rr = ref_ds.reshape(walk["s_norm"].shape).to(torch.float64)
            vs_ref["device_vs_reference"] = stats(dev_node["s_norm"], rr)
            vs_ref["float64walk_vs_reference"] = stats(walk["s_norm"], rr)
            vs_ref["device_vs_float64walk"] = stats(dev_node["s_norm"], walk["s_norm"])

        iso_sorted = sorted(iso, key=lambda x: -x["rel"])
        report["blocks"][str(blk)] = {
            "passes": len(recs[blk]), "ops": len(ops),
            "score_scale_applied_by_the_card": f_fwd,
            "score_scale_written": float(hd) ** -0.5,
            "isolation": iso_sorted,
            "isolation_worst": iso_sorted[:6],
            "walk": walk_rows,
            "vs_reference": vs_ref,
            "norms": {"s_in": float(s_norm.norm()), "z_in": float(z.norm()),
                      "dx": float(dx.norm())},
        }
        print(f"block {blk}: worst op " + json.dumps(iso_sorted[0]) if iso_sorted else "")
        print("   walk ds: " + json.dumps([r for r in walk_rows if r["node"] == "s_norm"]))
        if vs_ref:
            print("   vs ref: " + json.dumps(vs_ref))

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
