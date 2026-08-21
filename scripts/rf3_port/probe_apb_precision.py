#!/usr/bin/env python3
"""Why the trunk Pairformer's AttentionPairBias is 13x its bf16 ceiling.

`probe_pairformer_perop.py` names the op: teacher-forced on the reference's own input,
attention_pair_bias scores rel_rms 0.0278 against a 0.0021 ceiling while the other six ops
in the block sit at 1.3-3.2x. This splits the candidates by running the same op four ways
on the same input.

    production        bf16 storage, fp32 softmax reduction  -- what ships
    bf16_softmax      bf16 storage, bf16 softmax            -- is the fp32 reduction load-bearing here
    fp32              fp32 linears/norms/gate, bf16 SDPA core (fp32 SDPA is op-blocked)
    no_bias_scale     production with scale_pair_bias=False -- the convention A/B, live in THIS harness

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \
        python3 scripts/rf3_port/probe_apb_precision.py \
            --ckpt ~/rf3_ref_work/rf3_latest.ckpt --golden ~/rf3_ref_work/trunk_io_ligands.pt
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
APB_HEAD_DIM, APB_HEADS = 24, 16


def rel_rms(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def build_block(sd):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import PairformerBlock
    blk = PairformerBlock(c_s=C_S, c_z=C_Z, p_drop=0.0,
                          triangle_multiplication={"d_hidden": 128},
                          triangle_attention={"n_head": 4, "d_hidden": 32},
                          attention_pair_bias={"n_head": 16})
    blk.load_state_dict(sd, strict=False)
    return blk.eval()


def call_apb(sd, s_in, z_in, autocast):
    """Call the reference attention directly, teacher-forced, the way the device side is.

    Direct rather than hooked so the pair input can be substituted: with z replaced by
    zeros the pair bias becomes a per-head constant, which softmax is invariant to, so the
    arm isolates the q/k/v -> softmax -> AV -> gate -> proj_o path from the bias path.
    """
    blk = build_block(sd)
    if not autocast:
        blk.attention_pair_bias.force_bfloat16 = False
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = blk.attention_pair_bias(s_in.clone(), None, z_in.clone(),
                                      Beta_II=torch.tensor([0.0]))
    return out.detach().float()


def capture_apb(sd, s, z, autocast):
    blk = build_block(sd)
    if not autocast:
        blk.attention_pair_bias.force_bfloat16 = False
    got = {}

    def hook(_m, inputs, output):
        got["in"] = [i.detach().float().clone() if torch.is_tensor(i) else i for i in inputs]
        got["out"] = output.detach().float().clone()

    h = blk.attention_pair_bias.register_forward_hook(hook)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        blk(s.clone(), z.clone())
    h.remove()
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.remap import remap_pairformer_block
    from tt_bio.tenstorrent import AttentionPairBias, get_device

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()

    lo = capture_apb(sd, s, z, autocast=True)
    hi = capture_apb(sd, s, z, autocast=False)
    ceiling = rel_rms(hi["out"], lo["out"])

    rm = remap_pairformer_block(sd)
    attn_sd = {k[len("attention."):]: v for k, v in rm.items() if k.startswith("attention.")}

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    s_in, z_in = lo["in"][0], lo["in"][2]

    def tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

    def normed(dtype):
        return ttnn.layer_norm(
            tt(s_in, dtype),
            weight=tt(rm["pre_norm_s.weight"], dtype), bias=tt(rm["pre_norm_s.bias"], dtype),
            epsilon=1e-5, compute_kernel_config=cfg)

    arms = {}
    for name, kw, dtype in (
            ("production", dict(scale_pair_bias=True, fp32_softmax=True), ttnn.bfloat16),
            ("bf16_softmax", dict(scale_pair_bias=True, fp32_softmax=False), ttnn.bfloat16),
            ("fp32", dict(scale_pair_bias=True, fp32_softmax=True), ttnn.float32),
            ("no_bias_scale", dict(scale_pair_bias=False, fp32_softmax=True), ttnn.bfloat16)):
        op = AttentionPairBias(APB_HEAD_DIM, APB_HEADS, True, False, attn_sd, cfg,
                               dtype=dtype, **kw)
        got = op(normed(dtype), tt(z_in, dtype))
        got = torch.Tensor(ttnn.to_torch(got)).float().reshape(lo["out"].shape)
        e = rel_rms(got, lo["out"])
        arms[name] = {"pcc": round(pcc(got, lo["out"]), 7), "rel_rms": round(e, 6),
                      "x_ceiling": round(e / ceiling, 2)}

    # Same op, same s, but z replaced by zeros: the bias is then constant per head and
    # softmax is invariant to it, so what is left is the non-bias path.
    z_zero = torch.zeros_like(z_in)
    lo_z0 = call_apb(sd, s_in, z_zero, autocast=True)
    hi_z0 = call_apb(sd, s_in, z_zero, autocast=False)
    ceil_z0 = rel_rms(hi_z0, lo_z0)
    op0 = AttentionPairBias(APB_HEAD_DIM, APB_HEADS, True, False, attn_sd, cfg,
                            scale_pair_bias=True, fp32_softmax=True)
    got0 = op0(normed(ttnn.bfloat16), tt(z_zero))
    got0 = torch.Tensor(ttnn.to_torch(got0)).float().reshape(lo_z0.shape)
    e0 = rel_rms(got0, lo_z0)
    arms["z_zero"] = {"pcc": round(pcc(got0, lo_z0), 7), "rel_rms": round(e0, 6),
                      "ceiling": round(ceil_z0, 6),
                      "x_ceiling": round(e0 / ceil_z0, 2) if ceil_z0 else None,
                      "out_std": round(float(lo_z0.std()), 4)}
    # And the direct call on the REAL z, as a control that a direct call reproduces the
    # hooked number (it is the same op on the same input, so it has to).
    lo_direct = call_apb(sd, s_in, z_in, autocast=True)
    arms["_direct_vs_hooked_ref"] = round(rel_rms(lo_direct, lo["out"]), 8)

    rep = {"stack": args.stack, "tokens": int(z.shape[-2]),
           "ceiling": round(ceiling, 6),
           "out_std": round(float(lo["out"].std()), 4),
           "kq_norm": "query_layer_norm.weight" in attn_sd,
           "arms": arms}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
