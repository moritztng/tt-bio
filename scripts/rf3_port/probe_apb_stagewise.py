#!/usr/bin/env python3
"""Stage-level bisect of `AttentionPairBias` on RF3's teacher-forced pairformer input.

`probe_pairformer_perop.py` named the op: `attention_pair_bias` scores rel_rms 0.027823
against a 0.002071 ceiling, 13.43x, four times clear of the next worst op in the block,
and that 13.43x IS the composed s-track's 11.37x. `probe_apb_precision.py` then ruled out
the bias-scale convention, fp32 storage outside the attention core, the whole bias path
and tile padding. What is left has to be separated stage by stage.

This rebuilds tt-bio's `AttentionPairBias.__call__` out of individual ttnn ops and scores
every intermediate against the reference's own stage, teacher-forcing EVERY stage on the
reference's fp32 intermediate so a stage's number is its own and not its predecessor's.
The ceiling per stage is the reference's autocast-bf16 stage against its fp32 stage, i.e.
the same ceiling definition the per-op probe uses.

The reference chain is re-implemented here rather than hooked, because stages 5-8 are not
submodule boundaries. `--check` asserts the re-implementation reproduces the real module's
output in both arms before any stage is scored, so the references are trustworthy.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \
        python3 scripts/rf3_port/probe_apb_stagewise.py \
            --ckpt ~/rf3_ref_work/rf3_latest.ckpt \
            --golden ~/rf3_ref_work/trunk_io_ligands.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

C_S, C_Z = 384, 128
N_HEAD, HEAD_DIM = 16, 24
PADDED = 32


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def build_block(block_sd: dict):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import PairformerBlock
    blk = PairformerBlock(
        c_s=C_S, c_z=C_Z, p_drop=0.0,
        triangle_multiplication={"d_hidden": 128},
        triangle_attention={"n_head": 4, "d_hidden": 32},
        attention_pair_bias={"n_head": 16},
    )
    missing, unexpected = blk.load_state_dict(block_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"block weights mismatch: {len(missing)} / {len(unexpected)}")
    return blk.eval()


def apb_inputs(block_sd: dict, s: torch.Tensor, z: torch.Tensor):
    """The input the reference block hands its attention_pair_bias, and the module itself."""
    blk = build_block(block_sd)
    got = {}

    def hook(_m, inputs, output):
        got["in"] = [i.detach().clone() if torch.is_tensor(i) else i for i in inputs]
        got["out"] = output.detach().float().clone()

    h = blk.attention_pair_bias.register_forward_hook(hook)
    # The reference apb casts A_I to bf16 unconditionally (force_bfloat16), so it only runs
    # under autocast. Capture the input there, which is also the arm upstream actually runs.
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        blk(s.clone(), z.clone())
    h.remove()
    return got, blk.attention_pair_bias


def ref_stages(apb, a0: torch.Tensor, zz: torch.Tensor, bf16: bool) -> dict:
    """The reference forward, stage by stage, exactly as written in
    `AttentionPairBiasPairformerDeepspeed.forward`.

    `bf16=True` is what upstream runs (force_bfloat16 casts A_I, and the trunk runs under
    torch.autocast); `bf16=False` is the fp32 truth. Everything else is line-for-line.
    """
    st = {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        a = apb.ln_1(a0)
        if bf16:
            a = a.to(torch.bfloat16)
        st["a"] = a
        st["q"], st["k"], st["v"] = apb.to_q(a), apb.to_k(a), apb.to_v(a)
        st["b"] = apb.to_b(apb.ln_0(zz)) + torch.tensor([0.0])[..., None]
        st["g"] = apb.to_g(a)
        # the raw (unscaled) logits, then the reference's own scale
        st["logits"] = torch.einsum("...ihd,...jhd->...ijh", st["q"], st["k"])
        qs = st["q"] / torch.sqrt(torch.tensor(HEAD_DIM).to(st["q"].dtype))
        st["probs"] = torch.softmax(
            torch.einsum("...ihd,...jhd->...ijh", qs, st["k"]) + st["b"], dim=-2)
        st["o"] = torch.einsum("...ijh,...jhc->...ihc", st["probs"], st["v"])
        st["gated"] = (st["g"] * st["o"]).flatten(start_dim=-2)
        st["out"] = apb.to_a(st["gated"])
    return {k: v.float() for k, v in st.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS, remap_pairformer_block
    from tt_bio.tenstorrent import PairformerLayer, batched_matmul, get_device

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    if not sd:
        raise KeyError(f"no weights under {args.stack!r}")

    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()
    if args.crop:
        s, z = s[:, :args.crop], z[:, :args.crop, :args.crop]

    got, apb = apb_inputs(sd, s, z)
    a0, zz = got["in"][0].float(), got["in"][2].float()

    hi = ref_stages(apb, a0, zz, bf16=False)
    lo = ref_stages(apb, a0, zz, bf16=True)
    # the re-implementation has to be the module, or none of the stage references mean anything
    check = rel_rms(lo["out"], got["out"])
    if check > 1e-6:
        raise RuntimeError(f"re-implementation does not reproduce the module: rel_rms {check:.3e}")

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer = PairformerLayer(32, 4, HEAD_DIM, N_HEAD, True, remap_pairformer_block(sd), cfg,
                            **PAIRFORMER_FLAGS)
    op = layer.attention_pair_bias
    scale = HEAD_DIM ** -0.5

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    def host(x, shape=None):
        y = torch.Tensor(ttnn.to_torch(x)).float()
        return y.reshape(shape) if shape is not None else y

    def pad_heads(x):
        """[B,I,H,24] -> device layout [B,H,I,32] with zero pad lanes, as the port builds it."""
        b, i, h, c = x.shape
        out = torch.zeros(b, i, h, PADDED)
        out[..., :c] = x
        return out.permute(0, 2, 1, 3).contiguous()

    rows = []

    def score(name, device_out, ref_hi, ref_lo, note=""):
        ceil = rel_rms(ref_hi, ref_lo)
        e = rel_rms(device_out, ref_lo)
        rows.append({"stage": name, "rel_rms": round(e, 6), "ceiling": round(ceil, 6),
                     "x_ceiling": round(e / ceil, 2) if ceil else None,
                     "ref_std": round(float(ref_lo.std()), 4), "note": note})

    # 1 -- the pre-attention norm, which tt-bio hoists into PairformerLayer
    a_dev = ttnn.layer_norm(tt(a0), weight=layer.pre_norm_s_weight,
                            bias=layer.pre_norm_s_bias, epsilon=1e-5,
                            compute_kernel_config=cfg)
    score("1_ln_1", host(a_dev, hi["a"].shape), hi["a"], lo["a"])

    # 2 -- the fused qkv projection and the padded head split
    a_tt = tt(hi["a"])
    qkv = ttnn.linear(a_tt, op.qkv_weight, bias=op.qkv_bias,
                      compute_kernel_config=cfg)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.unsqueeze(qkv, 1), num_heads=N_HEAD, num_kv_heads=N_HEAD,
        transpose_k_heads=False)
    for nm, dv in (("q", q), ("k", k), ("v", v)):
        d = host(dv)[..., :HEAD_DIM]                        # [B,H,I,24]
        score(f"2_{nm}", d, hi[nm].permute(0, 2, 1, 3), lo[nm].permute(0, 2, 1, 3))
        pad = host(dv)[..., HEAD_DIM:]
        rows[-1]["pad_absmax"] = round(float(pad.abs().max()), 6)

    # 3 -- the sigmoid gate
    g_dev = ttnn.linear(a_tt, op.g_weight, compute_kernel_config=cfg)
    g_dev = ttnn.sigmoid(g_dev)
    score("3_gate", host(g_dev).reshape(lo["g"].flatten(start_dim=-2).shape),
          hi["g"].flatten(start_dim=-2), lo["g"].flatten(start_dim=-2))

    # 4 -- the pair bias. z_weight carries the sqrt(head_dim) pre-bake, so undo it to
    #      compare against the reference's unscaled B_IIH.
    zn = ttnn.layer_norm(tt(zz), weight=op.z_norm_weight, bias=op.z_norm_bias,
                         epsilon=1e-5, compute_kernel_config=cfg)
    zb = ttnn.linear(zn, op.z_weight, compute_kernel_config=cfg)
    bias_dev = ttnn.permute(zb, (0, 3, 1, 2))
    b_ref_hi, b_ref_lo = (t["b"].permute(0, 3, 1, 2) for t in (hi, lo))
    score("4_bias", host(bias_dev, b_ref_lo.shape) * (1.0 / op._bias_scale),
          b_ref_hi, b_ref_lo, note=f"bias_scale {op._bias_scale:.6f} undone on host")

    # 5 -- the raw q@k^T, on the reference's own fp32 q and k
    q_r, k_r = tt(pad_heads(hi["q"])), tt(pad_heads(hi["k"]))
    kt = ttnn.permute(k_r, (0, 1, 3, 2))
    lg = batched_matmul(q_r, kt, compute_kernel_config=cfg)
    lg_ref_hi, lg_ref_lo = (t["logits"].permute(0, 3, 1, 2) for t in (hi, lo))
    score("5_logits", host(lg, lg_ref_lo.shape), lg_ref_hi, lg_ref_lo,
          note=f"absmax {float(lg_ref_lo.abs().max()):.1f}")

    # 6 -- the production softmax tail, on the reference's own fp32 logits and bias
    from tt_bio.tenstorrent import _fp32_softmax_tail
    lg_r = tt(lg_ref_hi)
    bias_baked = tt(b_ref_hi * op._bias_scale)   # as the port's z_weight would produce it
    probs = _fp32_softmax_tail(lg_r, bias_baked, scale, 1.0 / op._bias_scale, None)
    p_ref_hi, p_ref_lo = (t["probs"].permute(0, 3, 1, 2) for t in (hi, lo))
    score("6_softmax", host(probs, p_ref_lo.shape), p_ref_hi, p_ref_lo,
          note="_fp32_softmax_tail, teacher-forced logits+bias")
    # and the same tail with the bias taken from the port's own device bias, to price the
    # pre-bake round-trip separately from the softmax itself
    probs2 = _fp32_softmax_tail(tt(lg_ref_hi), bias_dev, scale, 1.0 / op._bias_scale, None)
    score("6b_softmax_devbias", host(probs2, p_ref_lo.shape), p_ref_hi, p_ref_lo,
          note="same tail, port's own device-computed bias")

    # 7 -- probs @ v, on the reference's own fp32 probs and v
    o_dev = batched_matmul(tt(p_ref_hi), tt(pad_heads(hi["v"])), compute_kernel_config=cfg)
    o_ref_hi, o_ref_lo = (t["o"].permute(0, 2, 1, 3) for t in (hi, lo))
    score("7_pv", host(o_dev)[..., :HEAD_DIM], o_ref_hi, o_ref_lo)

    # 8 -- the gate multiply, in the port's flattened layout
    o_flat_hi = hi["o"].flatten(start_dim=-2)
    gated = ttnn.multiply(tt(o_flat_hi), tt(hi["g"].flatten(start_dim=-2)))
    score("8_gate_mul", host(gated, lo["gated"].shape), hi["gated"], lo["gated"])

    # 9 -- the output projection
    out_dev = ttnn.linear(tt(hi["gated"]), op.o_weight, compute_kernel_config=cfg)
    score("9_proj_o", host(out_dev, lo["out"].shape), hi["out"], lo["out"])

    # the whole op, unbisected, for the row this table has to explain
    whole = op(ttnn.layer_norm(tt(a0), weight=layer.pre_norm_s_weight,
                               bias=layer.pre_norm_s_bias, epsilon=1e-5,
                               compute_kernel_config=cfg), tt(zz))
    score("op_composed", host(whole, lo["out"].shape), hi["out"], lo["out"])

    rep = {"stack": args.stack, "tokens": int(zz.shape[-2]), "crop": args.crop,
           "reimpl_check_rel_rms": float(f"{check:.3e}"), "stages": rows}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
