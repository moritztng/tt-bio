"""AttentionPairBias under EXACT reference inputs (7XI5 cycle 2, block 44): pushes the
reference-computed pre_norm_s output and the reference mid-block z to the device op,
isolating the device AttentionPairBias from inherited input error.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_apb_exact.py \
    --query-json /tmp/p13_query_7XI5.json [--cycle 2] [--block 44]
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.environ.get("OF3_REF", "/tmp/of3-ref"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--cycle", type=int, default=2)
    ap.add_argument("--block", type=int, default=44)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3 as RefOpenFold3

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    query = next(iter(InferenceQuerySet.from_json(args.query_json).queries.values()))
    features = build_openfold3_features(query)
    batch = {k: v.unsqueeze(0) for k, v in features.items() if torch.is_tensor(v)}

    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref_model = RefOpenFold3(config=C).eval()
    ref_model.load_state_dict(sd, strict=False)

    cap = {"s_glue": [], "z_msa": []}

    def pf_pre(_m, _args, kwargs):
        cap["s_glue"].append(kwargs["s"].detach().clone())
        cap["z_msa"].append(kwargs["z"].detach().clone())

    ref_model.pairformer_stack.register_forward_pre_hook(pf_pre, with_kwargs=True)
    with torch.no_grad():
        ref_model.run_trunk(batch, num_cycles=args.cycle + 1)

    s_in, z_in = cap["s_glue"][args.cycle], cap["z_msa"][args.cycle]
    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]

    B = args.block
    blk_ref = ref_model.pairformer_stack.blocks[B]
    op_cap, z_mid = {}, {}

    h1 = blk_ref.attn_pair_bias.register_forward_hook(lambda _m, _a, out: op_cap.setdefault("d_attn", out.detach().clone()))
    h3 = blk_ref.pair_stack.register_forward_hook(lambda _m, _a, out: z_mid.setdefault("z", out.detach().clone()))
    s_b, z_b = s_in.clone(), z_in.clone()
    with torch.no_grad():
        for b in list(ref_model.pairformer_stack.blocks)[:B]:
            s_b, z_b = b(s=s_b, z=z_b, single_mask=token_mask.to(dtype=z_b.dtype),
                         pair_mask=pair_mask.to(dtype=s_b.dtype), chunk_size=None,
                         use_deepspeed_evo_attention=False,
                         use_triton_triangle_kernels=False,
                         use_cueq_triangle_kernels=False,
                         use_lma=False, inplace_safe=False, _mask_trans=True)
        blk_ref(s=s_b, z=z_b, single_mask=token_mask.to(dtype=z_b.dtype),
                pair_mask=pair_mask.to(dtype=s_b.dtype), chunk_size=None,
                use_deepspeed_evo_attention=False, use_triton_triangle_kernels=False,
                use_cueq_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True)
    h1.remove()
    h3.remove()
    d_attn_ref = op_cap["d_attn"][0]
    z_mid_ref = z_mid["z"][0]

    # reference pre-norm s (device applies pre_norm_s before the op)
    ln_w = sd[f"pairformer_stack.blocks.{B}.attn_pair_bias.layer_norm_a.weight"]
    ln_b = sd[f"pairformer_stack.blocks.{B}.attn_pair_bias.layer_norm_a.bias"]
    s_norm_ref = torch.nn.functional.layer_norm(s_b[0], (s_b.shape[-1],), ln_w, ln_b, 1e-5)

    import ttnn

    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    def ft(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    trunk = OF3Trunk(sd, ckc, num_cycles=1)
    blk = trunk.pairformer.blocks[B]
    n_token = int(s_in.shape[1])

    d_attn_d = blk.attention_pair_bias(ft(s_norm_ref.unsqueeze(0)),
                                       ft(z_mid_ref.unsqueeze(0)), seq_mask=None)
    d_attn_h = ttnn.to_torch(d_attn_d).float().reshape(n_token, -1)
    print(f"EXACT-INPUT device attn_pair_bias vs reference: pcc={pcc(d_attn_h, d_attn_ref):.6f} "
          f"(dev std {float(d_attn_h.std()):.2f} vs ref std {float(d_attn_ref.std()):.2f}, "
          f"ratio {float(d_attn_h.std() / d_attn_ref.std()):.4f})")


if __name__ == "__main__":
    main()
