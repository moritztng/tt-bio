"""Where inside the AE blocks does the device lose precision? For one golden case and one half, runs the
device stack block by block and, at every block, re-runs the same block in float64 from the DEVICE's
own input, reading every intermediate back. Each line: block, site, device error vs float64, and the
same site's bf16-weight floor (float64 with only the matrices rounded), both as max|d| / max|ref|
over the attended tokens.

    TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/rfd4/vae/debug_block.py --golden <case.pt> \
        --ckpt <ae.ckpt> --part decoder [--blocks 0,5,11]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--part", default="decoder")
    ap.add_argument("--blocks", default="")
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.rfd4 import ae_fold as AF
    from tt_bio.rfd4.autoencoder import RFD4AutoEncoder
    from tt_bio.token_axis import TILE, bucketed_width

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ae = RFD4AutoEncoder(sd, parts=(a.part,))
    m = ae.dev[a.part]
    f64 = AF.fold(sd, a.part, torch.float64)
    bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
    g = torch.load(a.golden, map_location="cpu", weights_only=False)
    batch = g["batch"]
    attend, _ = AF.ae_masks(batch)
    feats = (AF.encoder_features(batch, f64["emb"]) if a.part == "encoder"
             else AF.decoder_features(batch, f64["emb"], ae.tab["mask_token"]))
    B, N, _ = feats.shape
    Np = bucketed_width(N, TILE)
    att = torch.zeros(B, Np, dtype=torch.bool)
    att[:, :N] = attend
    x = torch.zeros(B, Np, AF.D, dtype=torch.float64)
    x[:, :N] = (feats.double() @ f64["in_w"].T) * attend[..., None]
    bias64 = AF.key_bias(att, torch.float64)
    mask = att[..., None].double()
    dev = m.device
    up = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    down = lambda t: ttnn.to_torch(t).double()  # noqa: E731
    bias = up(bias64.expand(B, AF.HEADS, Np, Np))
    kmean = up((att / att.sum(-1, keepdim=True).clamp_min(1)).double()[:, None, :])
    H, DH, D = AF.HEADS, AF.DH, AF.D
    want = {int(i) for i in a.blocks.split(",")} if a.blocks else set(range(AF.LAYERS))
    tok = att  # [B, Np]

    def err(t, r, rows=True):
        if rows:
            t, r = t[tok], r[tok]
        return float((t - r).abs().max() / r.abs().max().clamp_min(1e-30))

    def ref_block(L, x, c=None):
        """float64 block returning its intermediates (heads layout as the device's)."""
        b_, n, _ = x.shape
        o = {}
        o["h"] = AF._ln(AF._ln(x) * L["ln1_w"] + L["ln1_b"], L["node_w"], L["node_b"])
        q, k, v = (o["h"] @ L["qkv_w"].T + L["qkv_b"]).chunk(3, -1)
        q, k = AF._ln(q, L["q_ln_w"], L["q_ln_b"]), AF._ln(k, L["k_ln_w"], L["k_ln_b"])
        if c is not None:
            k = k - c
        q, k, v = (t.reshape(b_, n, H, DH).transpose(1, 2) for t in (q, k, v))
        o["q"], o["k"], o["v"] = q, k, v
        o["s"] = q @ k.transpose(-1, -2) * DH ** -0.5 + bias64
        o["p"] = torch.softmax(o["s"], -1)
        att_o = (o["p"] @ v).transpose(1, 2).reshape(b_, n, D)
        o["o"] = att_o
        o["og"] = torch.sigmoid(o["h"] @ L["g_w"].T + L["g_b"]) * att_o
        o["x2"] = x + o["og"] @ L["o_w"].T + L["o_b"]
        o["t"] = AF._ln(o["x2"]) * L["ln2_w"] + L["ln2_b"]
        u, gate = (o["t"] @ L["w1"].T).chunk(2, -1)
        o["ug"] = F.silu(gate) * u
        o["out"] = o["x2"] + o["ug"] @ L["w2"].T
        return o

    ckc = m.compute_kernel_config
    X = up(x)
    for i, (L, Ld) in enumerate(zip(f64["layers"], m.layers)):
        if i in want:
            xin = down(X)
            Lb = dict(L, **{k: bf(L[k]) for k in ("qkv_w", "g_w", "o_w", "w1", "w2")})
            d = {}
            # Mirrors AEHalf._block step by step (common-mode split, centred keys), reading every site back.
            h1 = m._ln(X, Ld["ln1_w"], Ld["ln1_b"])
            h0 = m._ln(h1, Ld["node_w"], Ld["node_b"])
            d["h"] = down(h0)
            h, ch = m._split_common(h0, kmean)
            q = m._ln(m._lin_cm(h, ch, Ld, "q"), Ld["q_ln_w"], Ld["q_ln_b"])
            k = m._ln(m._lin_cm(h, ch, Ld, "k"), Ld["k_ln_w"], Ld["k_ln_b"])
            kc = ttnn.matmul(kmean, k, dtype=ttnn.float32, compute_kernel_config=ckc)
            c64 = down(kc)  # the same constant on the reference side keeps "k" and "s" comparable
            k = ttnn.subtract(k, kc)
            v = m._lin_cm(h, ch, Ld, "v", row=False)
            vrow = m._row(ch, Ld["v_h"])
            d["v"] = down(ttnn.add(v, vrow)).reshape(B, Np, H, DH).transpose(1, 2)
            q, k, v = m._split_heads(ttnn.concat([q, k, v], dim=-1), H)
            d["q"], d["k"] = down(q), down(k)
            s = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), dtype=ttnn.float32, compute_kernel_config=ckc)
            s = ttnn.add(ttnn.multiply(s, DH ** -0.5), bias)
            d["s"] = down(s)
            p = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
            d["p"] = down(p)
            o = ttnn.add(m._merge_heads(ttnn.matmul(p, v, dtype=ttnn.float32, compute_kernel_config=ckc)), vrow)
            d["o"] = down(o)
            og = ttnn.multiply(o, ttnn.sigmoid(m._lin_cm(h, ch, Ld, "g")))
            d["og"] = down(og)
            ogc, cog = m._split_common(og, kmean)
            x2 = ttnn.add(X, m._lin_cm(ogc, cog, Ld, "o"))
            d["x2"] = down(x2)
            t = m._ln(x2, Ld["ln2_w"], Ld["ln2_b"])
            d["t"] = down(t)
            tc, ct = m._split_common(t, kmean)
            ug = ttnn.multiply(m._lin_cm(tc, ct, Ld, "w1a"), ttnn.silu(m._lin_cm(tc, ct, Ld, "w1g")))
            d["ug"] = down(ug)
            ugc, cug = m._split_common(ug, kmean)
            d["out"] = down(ttnn.add(x2, m._lin_cm(ugc, cug, Ld, "w2")))
            r, rf = ref_block(L, xin, c64), ref_block(Lb, xin, c64)
            for site in ("h", "q", "k", "v", "s", "p", "o", "og", "x2", "t", "ug", "out"):
                if site in ("q", "k", "v", "s", "p"):
                    sel = lambda z: z.transpose(1, 2)  # noqa: E731  token axis to dim 1 for the row mask
                    if site in ("s", "p"):
                        sel = lambda z: z.transpose(1, 2)[..., :N]  # noqa: E731
                    e_d, e_f = err(sel(d[site]), sel(r[site])), err(sel(rf[site]), sel(r[site]))
                else:
                    e_d, e_f = err(d[site], r[site]), err(rf[site], r[site])
                print(f"block {i:2d} {site:4s} dev {e_d:.3e} floor {e_f:.3e} ratio {e_d / max(e_f, 1e-30):6.1f}", flush=True)
        X = m._block(Ld, X, bias, kmean)
    xf = down(X)
    ref = x.clone()
    for L in f64["layers"]:
        ref = AF.layer_forward(L, ref, bias64, mask)
    print("stack out", err(xf, ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
