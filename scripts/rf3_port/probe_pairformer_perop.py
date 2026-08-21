#!/usr/bin/env python3
"""Per-op parity inside one RF3 Pairformer block, teacher-forced on real inputs.

The block scores s rel_rms 0.021 / z 0.004 against a torch golden whose own bf16 spread
is 0.0019 / 0.00034, so both tracks sit ~11x their ceiling. `probe_bf16_throughout.py`
rules out the "ttnn carries bf16 activations, autocast does not" explanation (bf16 storage
throughout costs 0.95x, not 11x), so the gap is in an op. This names it.

Every op is fed the input the REFERENCE gave it (hooks on the vendored block during a real
forward) and scored against the output the reference produced, with the ceiling measured on
that same input as autocast-vs-fp32. Teacher-forcing is the point: composed error tells you
a block is off, per-op error tells you which one.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \
        python3 scripts/rf3_port/probe_pairformer_perop.py \
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
DIMS = (32, 4, 24, 16)   # tri_att head_dim / heads, attention-pair-bias head_dim / heads

#: reference submodule -> the tt-bio child of `PairformerLayer` that ports it
OPS = [
    ("tri_mul_outgoing", "triangle_multiplication_start"),
    ("tri_mul_incoming", "triangle_multiplication_end"),
    ("tri_attn_start", "triangle_attention_start"),
    ("tri_attn_end", "triangle_attention_end"),
    ("z_transition", "transition_z"),
    ("attention_pair_bias", "attention_pair_bias"),
    ("s_transition", "transition_s"),
]


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


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


def capture(block_sd: dict, s: torch.Tensor, z: torch.Tensor, autocast: bool) -> dict:
    """Run the reference block once, keeping every op's input and output."""
    blk = build_block(block_sd)
    if not autocast:
        blk.attention_pair_bias.force_bfloat16 = False
    got: dict[str, dict] = {}

    def hook(name):
        def fn(_m, inputs, output):
            got[name] = {"in": [i.detach().float().clone() if torch.is_tensor(i) else i
                                for i in inputs],
                         "out": output.detach().float().clone()}
        return fn

    handles = [getattr(blk, ref).register_forward_hook(hook(ref)) for ref, _ in OPS]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        s_out, z_out = blk(s.clone(), z.clone())
    for h in handles:
        h.remove()
    got["_block"] = {"out": (s_out.float(), z_out.float())}
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True, help="capture_trunk_io.py output")
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--crop", type=int, default=0,
                    help="crop the captured input to N tokens. Every capture fixture is "
                         "8-53 tokens, i.e. 17-75%% tile padding on the device, and a pad "
                         "lane is not a lane both attention paths treat the same way. A "
                         "crop to a multiple of 32 is the same real input with the padding "
                         "gone, and the reference is recomputed on it.")
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS, remap_pairformer_block
    from tt_bio.tenstorrent import PairformerLayer, get_device

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

    lo = capture(sd, s, z, autocast=True)    # what upstream runs: the golden
    hi = capture(sd, s, z, autocast=False)   # fp32 truth, for the ceiling

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer = PairformerLayer(*DIMS, True, remap_pairformer_block(sd), cfg,
                            **PAIRFORMER_FLAGS)

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    rows = []
    for ref_name, tt_name in OPS:
        op = getattr(layer, tt_name)
        ref_out = lo[ref_name]["out"]
        x = lo[ref_name]["in"][0]
        if ref_name == "attention_pair_bias":
            # The reference's apb norms its own input (ln_1); tt-bio hoists that into
            # PairformerLayer as pre_norm_s and its AttentionPairBias does NOT norm. Feed
            # the normed input, or the op scores rel_rms 579 and reads as a broken port.
            x = ttnn.layer_norm(tt(x), weight=layer.pre_norm_s_weight,
                                bias=layer.pre_norm_s_bias, epsilon=1e-5,
                                compute_kernel_config=cfg)
            got = op(x, tt(lo[ref_name]["in"][2]))
        else:
            got = op(tt(x))
        got = torch.Tensor(ttnn.to_torch(got)).float().reshape(ref_out.shape)
        ceil = rel_rms(hi[ref_name]["out"], ref_out)
        e = rel_rms(got, ref_out)
        rows.append({"op": ref_name, "shape": list(ref_out.shape),
                     "pcc": round(pcc(got, ref_out), 7),
                     "rel_rms": round(e, 6), "ceiling": round(ceil, 6),
                     "x_ceiling": round(e / ceil, 2) if ceil else None,
                     "out_std": round(float(ref_out.std()), 4),
                     "in_std": round(float(lo[ref_name]["in"][0].std()), 4)})

    # The composed block, for the same run, so the per-op rows can be read against it.
    s_dev, z_dev = layer(tt(s), tt(z))
    s_lo, z_lo = lo["_block"]["out"]
    s_hi, z_hi = hi["_block"]["out"]
    composed = {
        "s_rel_rms": round(rel_rms(torch.Tensor(ttnn.to_torch(s_dev)).reshape(s_lo.shape),
                                   s_lo), 6),
        "z_rel_rms": round(rel_rms(torch.Tensor(ttnn.to_torch(z_dev)).reshape(z_lo.shape),
                                   z_lo), 6),
        "s_ceiling": round(rel_rms(s_hi, s_lo), 6),
        "z_ceiling": round(rel_rms(z_hi, z_lo), 6),
    }
    composed["s_x_ceiling"] = round(composed["s_rel_rms"] / composed["s_ceiling"], 2)
    composed["z_x_ceiling"] = round(composed["z_rel_rms"] / composed["z_ceiling"], 2)

    rep = {"stack": args.stack, "tokens": int(z.shape[-2]), "crop": args.crop,
           "per_op_teacher_forced": rows, "composed": composed}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
