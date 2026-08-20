"""Does the 848-token triangle-attention SDPA attend to its 16 tile-pad keys?

848 is not a multiple of 32, so a [*, *, 848, *] tensor's padded_shape is 864 and 16 key rows of
tile padding sit inside the last tile. `sdpa_generic.plan` reads `Sk` off `padded_shape`, so the
transcription's `valid_Skt = div_up(864, 32) = 27` covers every tile including the partial one --
nothing tells it that rows 848..863 are padding. If the stock op does the same, then every
triangle-attention softmax on this stage sums 864 keys, 16 of which are zeros with a zero bias, and
each carries `exp(0)` of the mass.

Two references, both in fp32 on host:
  ref848  softmax over the 848 real keys, which is what the maths wants
  ref864  softmax over 864 keys with k/v zero and bias zero in the last 16, which is what a kernel
          that trusts padded_shape computes
Whichever a device arm matches says what that arm actually did. Run at B=8 so the reference is
cheap; the split changes with B but the reduction the softmax performs does not.
"""
import argparse, json, os, sys
from pathlib import Path

H, D, S = 4, 32, 848


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(a.tree.resolve()))
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as F

    B = a.batch
    torch.manual_seed(0)
    dev = T.get_device()
    scale = D ** -0.5
    padded = T._padded_sdpa_len(S)

    hq = torch.randn(B, H, S, D).to(torch.bfloat16)
    hk = torch.randn(B, H, S, D).to(torch.bfloat16)
    hv = torch.randn(B, H, S, D).to(torch.bfloat16)
    hb = torch.randn(1, H, S, S).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    q, k, v, bias = up(hq), up(hk), up(hv), up(hb)

    sdpa = torch.nn.functional.scaled_dot_product_attention
    ref848 = sdpa(hq.float(), hk.float(), hv.float(),
                  attn_mask=hb.float().expand(B, -1, -1, -1), scale=scale)
    pk = torch.zeros(B, H, padded, D); pk[:, :, :S] = hk.float()
    pv = torch.zeros(B, H, padded, D); pv[:, :, :S] = hv.float()
    pb = torch.zeros(1, H, S, padded); pb[:, :, :, :S] = hb.float()
    ref864 = sdpa(hq.float(), pk, pv, attn_mask=pb.expand(B, -1, -1, -1), scale=scale)
    # ttnn applies `scale` to (QK^T + mask), not to QK^T alone: `TriangleAttention.__init__` sets
    # `_bias_scale = self.scale` and pre-multiplies the bias projection weight by it
    # (tenstorrent.py:3249, 3265), which only cancels if the op scales the mask too. So the
    # reference the fold actually wants is softmax(scale*QK^T + scale*mask).
    refconv = sdpa(hq.float(), hk.float(), hv.float(),
                   attn_mask=hb.float().expand(B, -1, -1, -1) * scale, scale=scale)

    rec = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(T.COMPUTE_GRID_MAIN), "tokens": S, "padded": padded, "batch": B,
           "pad_keys": padded - S, "arms": []}
    # How far apart the two references are at all: if they agree, this test cannot distinguish.
    d = ref864 - ref848
    rec["ref864_vs_ref848"] = {
        "rmsd_over_std": round(float(d.pow(2).mean().sqrt() / ref848.std()), 6),
        "max_abs": round(float(d.abs().max()), 6)}
    dc = refconv - ref848
    rec["refconv_vs_ref848"] = {
        "rmsd_over_std": round(float(dc.pow(2).mean().sqrt() / ref848.std()), 6),
        "max_abs": round(float(dc.abs().max()), 6)}

    cases = [("stock_q288_k256", "stock", 288, 256), ("stock_q864_k96", "stock", 864, 96),
             ("fused_q288_k864", "fused", 288, 864), ("fused_q288_k288", "fused", 288, 288),
             ("fused_q288_k96", "fused", 288, 96)]
    for label, path, qc, kc in cases:
        e = {"arm": label, "path": path, "q_chunk": qc, "k_chunk": kc}
        try:
            if path == "fused":
                o = F.sdpa(q, k, v, bias, scale, qc, kc)
                e["served"] = o is not None
            else:
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, is_causal=False, scale=scale,
                    program_config=T._sdpa_program_config(qc, kc))
                e["served"] = True
            if o is not None:
                ot = ttnn.to_torch(o).float()
                for tag, ref in (("848", ref848), ("864", ref864), ("conv", refconv)):
                    dd = ot - ref
                    e[f"rmsd_over_std_vs_ref{tag}"] = round(
                        float(dd.pow(2).mean().sqrt() / ref.std()), 6)
                ttnn.deallocate(o)
                del ot
        except Exception as exc:                                        # noqa: BLE001
            e["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["arms"].append(e)
        print(json.dumps(e), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({"ref864_vs_ref848": rec["ref864_vs_ref848"],
                      "refconv_vs_ref848": rec["refconv_vs_ref848"]}), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
