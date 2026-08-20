#!/usr/bin/env python3
"""Bisect the Pairformer s-track: attention_pair_bias vs s_transition.

The confidence head showed post_pairformer_s at rel_rms 0.0854 while z is 0.0079, and
the trunk Pairformer has carried an unexplained 11x s-track gap for several passes.
The s path inside a block is just two ops, so score each on real captured inputs and
teacher-force the other from the reference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.confidence_head.pairformer.0."
APB_HEAD_DIM, APB_HEADS = 24, 16


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_conf.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.remap import remap_pairformer_block
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import AttentionPairBias, Transition, get_device

    cap = torch.load(args.capture, weights_only=False)
    s_trunk, z_trunk = cap["in"][1].float(), cap["in"][2].float()
    s = s_trunk.reshape(1, *s_trunk.shape)
    z = z_trunk.reshape(1, *z_trunk.shape)

    net, _ = load_reference(args.ckpt, num_steps=2)
    blk = net.confidence_head.pairformer[0]
    beta = torch.tensor([0.0])

    def ref_pieces(bf16):
        blk.attention_pair_bias.force_bfloat16 = bf16
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            a = blk.attention_pair_bias(s, None, z, Beta_II=beta).float()
            t = blk.s_transition(s).float()
        blk.attention_pair_bias.force_bfloat16 = True
        return a, t

    hi_a, hi_t = ref_pieces(False)
    lo_a, lo_t = ref_pieces(True)

    sd = {k[len(PREFIX):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(PREFIX)}
    rm = remap_pairformer_block(sd)
    attn_sd = {k[len("attention."):]: v for k, v in rm.items()
               if k.startswith("attention.")}
    trans_sd = {k[len("transition_s."):]: v for k, v in rm.items()
                if k.startswith("transition_s.")}

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    # PairformerLayer applies pre_norm_s (RF3's ln_1, which the remap maps to
    # pre_norm_s) BEFORE the attention -- tt-bio's AttentionPairBias does not norm its
    # own input. Calling the attention directly on raw s skips it, and s_trunk has
    # std 514, which is worth rel_rms 579 and looks exactly like a broken port.
    # A/B the bias-scaling convention rather than reasoning about it. tt-bio's non-atom
    # attention computes softmax((q@k^T + bias) * head_dim**-0.5), so the bias is scaled
    # along with QK; RF3 divides Q first and adds the bias UNSCALED. scale_pair_bias=True
    # pre-multiplies z_weight by sqrt(head_dim) to compensate. The trunk port was built
    # with False, and the s-track has been unexplained ever since.
    def s_normed_for(_arm=None):
        return ttnn.layer_norm(
            tt(s),
            weight=ttnn.from_torch(rm["pre_norm_s.weight"], layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16),
            bias=ttnn.from_torch(rm["pre_norm_s.bias"], layout=ttnn.TILE_LAYOUT,
                                 device=dev, dtype=ttnn.bfloat16),
            epsilon=1e-5, compute_kernel_config=cfg)

    arms = {}
    for arm in (False, True):
        a = AttentionPairBias(APB_HEAD_DIM, APB_HEADS, True, False, attn_sd, cfg,
                              scale_pair_bias=arm, fp32_softmax=True)
        g = torch.Tensor(ttnn.to_torch(a(s_normed_for(arm), tt(z)))).float().reshape(hi_a.shape)
        arms[f"scale_pair_bias={arm}"] = {
            "pcc": round(pcc(g, lo_a), 7), "rel_rms": round(rel_rms(g, lo_a), 6),
            "x_ceiling": round(rel_rms(g, lo_a) / rel_rms(lo_a, hi_a), 2)}
    attn = AttentionPairBias(APB_HEAD_DIM, APB_HEADS, True, False, attn_sd, cfg,
                             scale_pair_bias=False, fp32_softmax=True)
    s_normed = ttnn.layer_norm(
        tt(s),
        weight=ttnn.from_torch(rm["pre_norm_s.weight"], layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16),
        bias=ttnn.from_torch(rm["pre_norm_s.bias"], layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16),
        epsilon=1e-5, compute_kernel_config=cfg)
    got_a = torch.Tensor(ttnn.to_torch(attn(s_normed, tt(z)))).float().reshape(hi_a.shape)
    tr = Transition(trans_sd, cfg)
    got_t = torch.Tensor(ttnn.to_torch(tr(tt(s)))).float().reshape(hi_t.shape)

    rows = []
    for name, got, hi, lo in (("attention_pair_bias", got_a, hi_a, lo_a),
                              ("s_transition", got_t, hi_t, lo_t)):
        ceil = rel_rms(lo, hi)
        e = rel_rms(got, hi if False else lo)
        rows.append({"op": name, "pcc": round(pcc(got, lo), 7),
                     "rel_rms": round(e, 6), "ceiling": round(ceil, 6),
                     "x_ceiling": round(e / ceil, 2) if ceil else None})
    print(json.dumps({"tokens": int(s.shape[1]), "scores": rows,
                      "bias_scaling_ab": arms}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
