#!/usr/bin/env python3
"""Score RF3's confidence head, ceiling measured in the same run, wrong variants too."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.confidence_head."


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
    from tt_bio.rf3.confidence_head import ConfidenceHead, predicted_distance_onehot
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    s_inputs, s_trunk, z_trunk, x_pred, seq, rep_atoms = cap["in"][:6]
    want = cap["out"]

    net, _ = load_reference(args.ckpt, num_steps=2)
    ref = net.confidence_head

    # The pairformer blocks pin their own precision (force_bfloat16=True inside
    # AttentionPairBias*), so autocast alone leaves both arms in bf16 and the ceiling
    # comes out as zero -- the same wrinkle the token DiT had.
    def run(bf16):
        for blk in ref.pairformer:
            blk.attention_pair_bias.force_bfloat16 = bf16
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            out = {k: v.float() for k, v in
                   ref(s_inputs, s_trunk, z_trunk, x_pred, seq, rep_atoms).items()}
        for blk in ref.pairformer:
            blk.attention_pair_bias.force_bfloat16 = True
        return out

    hi, lo = run(False), run(True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    w = {k[len(PREFIX):]: v.float()
         for k, v in torch.load(args.ckpt, map_location="cpu",
                                weights_only=False)["model"].items()
         if k.startswith(PREFIX)}

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    d1h = predicted_distance_onehot(x_pred, rep_atoms)
    head = ConfidenceHead(w, cfg)
    got = head(tt(s_inputs.reshape(1, *s_inputs.shape)),
               tt(s_trunk.reshape(1, *s_trunk.shape)),
               tt(z_trunk.reshape(1, *z_trunk.shape)), tt(d1h))

    # Bisect: is the gap in the pairformer stack or in the final norms/predictors?
    # Hook the reference's last pairformer block and compare its s/z against ours.
    post = {}
    h = ref.pairformer[-1].register_forward_hook(
        lambda _m, _i, o: post.update(s=o[0].detach().float(), z=o[1].detach().float()))
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            ref(s_inputs, s_trunk, z_trunk, x_pred, seq, rep_atoms)
    finally:
        h.remove()
    my_s, my_z = head.embed(tt(s_inputs.reshape(1, *s_inputs.shape)),
                            tt(s_trunk.reshape(1, *s_trunk.shape)),
                            tt(z_trunk.reshape(1, *z_trunk.shape)), tt(d1h))
    my_s, my_z = head.pairformer(my_s, my_z)
    stack = {}
    for nm, mine, ref_t in (("s", my_s, post["s"]), ("z", my_z, post["z"])):
        g = torch.Tensor(ttnn.to_torch(mine)).float().reshape(ref_t.shape)
        stack[f"post_pairformer_{nm}"] = {
            "pcc": round(pcc(g, ref_t), 7), "rel_rms": round(rel_rms(g, ref_t), 6)}

    rows = []
    for k in sorted(want):
        g = torch.Tensor(ttnn.to_torch(got[k])).float().reshape(want[k].shape)
        ceil = rel_rms(lo[k].reshape(want[k].shape), hi[k].reshape(want[k].shape))
        e = rel_rms(g, want[k])
        rows.append({"tensor": k, "shape": list(want[k].shape),
                     "pcc": round(pcc(g, want[k]), 7), "rel_rms": round(e, 6),
                     "ceiling": round(ceil, 6),
                     "x_ceiling": round(e / ceil, 2) if ceil else None})

    # The plausible wrong variant is a feature-dimension norm instead of the global one.
    # It cannot be produced by flipping layer_norm_along_feature_dimension, because that
    # branch is BROKEN upstream: `normalized_shape=(x.shape[-1])` is an int, not a tuple,
    # and torch rejects it. So the whole-tensor norm is the only reachable behaviour and
    # there is no ambiguity about which to implement. The input-level cost of getting it
    # wrong is measured separately in probe_confidence_layernorm.py
    # (pcc 0.9476 on Z_trunk_II, with an identical std to six figures).
    note = ("layer_norm_along_feature_dimension=True is dead code upstream: "
            "normalized_shape=(x.shape[-1]) is an int, torch requires a tuple")

    print(json.dumps({"scores": rows, "bisect": stack, "alt_branch": note}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
