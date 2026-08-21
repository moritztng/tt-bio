#!/usr/bin/env python3
"""On-device parity for RF3 -> tt-bio Pairformer block 0.

Runs RF3's torch `PairformerBlock` and tt-bio's shared `Pairformer` on the same
seeded (s, z) with the same checkpoint weights, and reports PCC on both tracks.

Dims match OpenFold3's: triangle attention 32x4, attention-pair-bias 24x16. RF3
pre-scales q and adds the pair bias unscaled (`attention.py::_forward_vanilla`),
which is the OF3 convention, so `scale_pair_bias=False`.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rf3-port-p1-parity \\
        python scripts/rf3_port/parity_pairformer.py --ckpt /path/to/rf3_latest.ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

STACK = "shadow.recycler.pairformer_stack."
C_S, C_Z = 384, 128
DIMS = (32, 4, 24, 16)  # tri_att head_dim / heads, apb head_dim / heads


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a * b).sum() / denom) if denom else float("nan")


def torch_golden(block_sd: dict, s: torch.Tensor, z: torch.Tensor, autocast: bool = True):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import PairformerBlock

    blk = PairformerBlock(
        c_s=C_S, c_z=C_Z, p_drop=0.0,
        triangle_multiplication={"d_hidden": 128},
        triangle_attention={"n_head": 4, "d_hidden": 32},
        attention_pair_bias={"n_head": 16},
    )
    missing, unexpected = blk.load_state_dict(block_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"block weights mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected {(missing or unexpected)[:3]}")
    blk.eval()
    # RF3's attention-pair-bias force-casts its input to bf16 while the weights stay
    # fp32, so the block only runs under autocast -- which is how upstream runs it
    # (Lightning sets bf16 AMP). The golden is therefore a bf16 golden, matching the
    # device side rather than an fp32 ideal neither of them computes.
    #
    # For the fp32 control that force-cast has to come off too, otherwise the block
    # dies on BFloat16 weights against float activations. That control is not a
    # reference, it is the answer to "how much of the gap is bf16 at all".
    if not autocast:
        blk.attention_pair_bias.force_bfloat16 = False
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        s_out, z_out = blk(s, z)
    return s_out.float(), z_out.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--golden", help="real trunk I/O from capture_trunk_io.py; "
                                     "without it the input is synthetic N(0,1)")
    ap.add_argument("--crop", type=int, default=0,
                    help="crop the golden input to N tokens. Every captured fixture is 8-53 "
                         "tokens, i.e. 17-75%% tile padding on the device, and a padded row is "
                         "not a row both attention paths compute the same way. A crop to a "
                         "multiple of 32 is the same real trunk input with the padding gone, and "
                         "the torch golden is recomputed on it, so it stays a real measurement.")
    args = ap.parse_args()

    import ttnn

    from tt_bio.rf3.remap import PAIRFORMER_FLAGS, remap_pairformer_block
    from tt_bio.tenstorrent import Pairformer, get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    pre = f"{STACK}0."
    block_sd = {k[len(pre):]: v.float() for k, v in sd.items() if k.startswith(pre)}
    if not block_sd:
        print(f"no weights at {pre}")
        return 1

    if args.golden:
        # Real operating point: what the block actually sees mid-trunk. Synthetic
        # N(0,1) drives this block far off-manifold (output std 735 against ~79 for
        # real input), and torch's own bf16 only reaches 0.982 against its fp32
        # there, so the synthetic number bounds the port's apparent quality rather
        # than measuring it.
        gold = torch.load(args.golden, weights_only=False)
        s, z = gold["in"]
        # The reference runs the trunk unbatched, so the hook captures [I, C] and
        # [I, I, C]; both sides here want a leading batch dim.
        s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
        z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()
        if args.crop:
            s, z = s[:, :args.crop], z[:, :args.crop, :args.crop]
        N = z.shape[-2]
    else:
        torch.manual_seed(args.seed)
        N = args.n
        s = torch.randn(1, N, C_S)
        z = torch.randn(1, N, N, C_Z)

    s_ref, z_ref = torch_golden(block_sd, s.clone(), z.clone())
    s_f32, z_f32 = torch_golden(block_sd, s.clone(), z.clone(), autocast=False)

    remapped = {f"layers.0.{k}": v for k, v in remap_pairformer_block(block_sd).items()}

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    # transpose_bias=False: RF3 builds the ending pair bias from the un-transposed
    # tensor. With the default the block scores z_pcc 0.82 instead of 0.99.
    # PAIRFORMER_FLAGS, not a hand-copied subset: this harness scored a configuration the
    # model does not run for as long as the two lists were kept separately, and missed both
    # gated_move and accurate_softmax that way.
    pf = Pairformer(1, *DIMS, True, remapped, cfg, **PAIRFORMER_FLAGS)

    def to_tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    s_out, z_out = pf(to_tt(s), to_tt(z))
    s_dev = torch.Tensor(ttnn.to_torch(s_out)).float().reshape(s_ref.shape)
    z_dev = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(z_ref.shape)

    rep = {
        "input": "real-trunk" if args.golden else "synthetic-N(0,1)",
        "tokens": N,
        "tile_pad_frac": round((-N % 32) / (N + (-N % 32)), 4),
        "s_pcc": round(pcc(s_dev, s_ref), 6),
        "z_pcc": round(pcc(z_dev, z_ref), 6),
        # The bf16 ceiling on this same input: how well torch itself does against
        # its own fp32. A device number at or above this is as good as bf16 allows.
        "s_ceiling_cpu_bf16_vs_fp32": round(pcc(s_ref, s_f32), 6),
        "z_ceiling_cpu_bf16_vs_fp32": round(pcc(z_ref, z_f32), 6),
        "s_ref_std": round(float(s_ref.std()), 4),
        "z_ref_std": round(float(z_ref.std()), 4),
        # PCC differences near 1.0 are unreadable and PCC is blind to a few entries
        # going badly wrong. A missing fp32_softmax showed up as PCC 0.9928 on the
        # template embedder while carrying 12% relative RMS.
        "s_rel_rms_device": round(
            float((s_dev - s_ref).pow(2).mean().sqrt() / s_ref.std()), 6),
        "z_rel_rms_device": round(
            float((z_dev - z_ref).pow(2).mean().sqrt() / z_ref.std()), 6),
        "s_rel_rms_reference": round(
            float((s_ref - s_f32).pow(2).mean().sqrt() / s_f32.std()), 6),
        "z_rel_rms_reference": round(
            float((z_ref - z_f32).pow(2).mean().sqrt() / z_f32.std()), 6),
        # The same device output against the FP32 golden. The bf16 golden above is the right
        # target for "does the port reproduce what upstream runs", and the wrong one for
        # "which of two device paths is more accurate": a path that mimics torch's autocast
        # rounding scores better against it than a path that is simply closer to the truth.
        # Reporting both is what separates those two questions.
        "s_rel_rms_device_vs_fp32": round(
            float((s_dev - s_f32).pow(2).mean().sqrt() / s_f32.std()), 6),
        "z_rel_rms_device_vs_fp32": round(
            float((z_dev - z_f32).pow(2).mean().sqrt() / z_f32.std()), 6),
        "s_pcc_vs_fp32": round(pcc(s_dev, s_f32), 6),
        "z_pcc_vs_fp32": round(pcc(z_dev, z_f32), 6),
    }
    # Which attention path actually ran. Without this a declined fused arm is bit-identical to the
    # materialised one and reads as a clean A/A rather than as "the lever never fired"
    # (`two-level-optin-ab-arm-and-page-provenance-drop`). The block routes qkv through L1 at some
    # sizes, and the fused kernel wants DRAM operands, so declining is size-conditioned and silent.
    import tt_bio.tenstorrent as _T
    rep["fused_hifi_enabled"] = _T._TRIATT_FUSED_HIFI
    rep["fused_hifi_stats"] = dict(_T.TRIATT_FUSED_HIFI_STATS)
    rep["fused_kernel_rejects"] = {str(k): v for k, v in _T._triatt_sdpa.REJECTS.items()}
    rep["s_at_ceiling"] = rep["s_pcc"] >= rep["s_ceiling_cpu_bf16_vs_fp32"] - 0.002
    rep["z_at_ceiling"] = rep["z_pcc"] >= rep["z_ceiling_cpu_bf16_vs_fp32"] - 0.002
    rep["verdict"] = (
        "PASS" if min(rep["s_pcc"], rep["z_pcc"]) > 0.98
        else "AT_BF16_CEILING" if rep["s_at_ceiling"] and rep["z_at_ceiling"]
        else "GAP"
    )
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] in ("PASS", "AT_BF16_CEILING") else 1


if __name__ == "__main__":
    sys.exit(main())
