"""7XI5 s-track per-OP bisect inside pairformer block 44 (largest per-block drop).
Captures the reference block's attention-pair-bias delta and single-transition delta
via hooks, then replicates the device block's z-track + s-track manually and PCCs
each intermediate.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_pf_op_bisect.py \
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

    s_in = cap["s_glue"][args.cycle]
    z_in = cap["z_msa"][args.cycle]
    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]

    B = args.block
    blk_ref = ref_model.pairformer_stack.blocks[B]
    ref_blocks = list(ref_model.pairformer_stack.blocks)
    op_cap = {}

    def mk(name):
        def hook(_m, _args, out):
            op_cap[name] = out.detach().clone()
        return hook

    h1 = blk_ref.attn_pair_bias.register_forward_hook(mk("d_attn"))
    h2 = blk_ref.single_transition.register_forward_hook(mk("d_trans"))
    # run blocks 0..B manually to capture the exact block-B inputs and op deltas
    s, z = s_in.clone(), z_in.clone()
    with torch.no_grad():
        for i, b in enumerate(ref_blocks[: B + 1]):
            s, z = b(s=s, z=z, single_mask=token_mask.to(dtype=z.dtype),
                     pair_mask=pair_mask.to(dtype=s.dtype), chunk_size=None,
                     use_deepspeed_evo_attention=False,
                     use_triton_triangle_kernels=False, use_cueq_triangle_kernels=False,
                     use_lma=False, inplace_safe=False, _mask_trans=True)
            if i == B:
                s_blk_in = op_cap  # deltas captured during this call
    h1.remove()
    h2.remove()
    d_attn_ref = op_cap["d_attn"]
    d_trans_ref = op_cap["d_trans"]
    print(f"reference block {B}: |d_attn| std {float(d_attn_ref.std()):.2f}, "
          f"|d_trans| std {float(d_trans_ref.std()):.2f}, "
          f"s_in std {float(s_in.std()):.2f} -> s_out std {float(s.std()):.2f}")

    # need the block-B INPUT states: rerun to capture
    s_b, z_b = s_in.clone(), z_in.clone()
    with torch.no_grad():
        for b in ref_blocks[:B]:
            s_b, z_b = b(s=s_b, z=z_b, single_mask=token_mask.to(dtype=z_b.dtype),
                         pair_mask=pair_mask.to(dtype=s_b.dtype), chunk_size=None,
                         use_deepspeed_evo_attention=False,
                         use_triton_triangle_kernels=False,
                         use_cueq_triangle_kernels=False,
                         use_lma=False, inplace_safe=False, _mask_trans=True)
    # reference intermediates inside block B (recompute from deltas)
    # z after pair_stack within block B is what attn_pair_bias consumed; capture via hook
    z_mid = {}
    h3 = blk_ref.pair_stack.register_forward_hook(lambda _m, _a, out: z_mid.setdefault("z", out.detach().clone()))
    with torch.no_grad():
        blk_ref(s=s_b.clone(), z=z_b.clone(), single_mask=token_mask.to(dtype=z_b.dtype),
                pair_mask=pair_mask.to(dtype=s_b.dtype), chunk_size=None,
                use_deepspeed_evo_attention=False, use_triton_triangle_kernels=False,
                use_cueq_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True)
    h3.remove()
    z_mid_ref = z_mid["z"]

    # ---- device: replicate block B with capture ---------------------------
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

    s_d, z_d = ft(s_b), ft(z_b)
    # z-track (device PairformerLayer order)
    for op in (blk.triangle_multiplication_start, blk.triangle_multiplication_end,
               blk.triangle_attention_start, blk.triangle_attention_end):
        zu = op(z_d, None)
        z_d = ttnn.add_(z_d, zu)
        ttnn.deallocate(zu)
    zu = blk.transition_z(z_d)
    z_d = ttnn.add_(z_d, zu)
    ttnn.deallocate(zu)
    z_mid_h = ttnn.to_torch(z_d).float().reshape(n_token, n_token, -1)
    print(f"z entering s-track:      pcc={pcc(z_mid_h, z_mid_ref[0]):.6f}")

    s_norm = ttnn.layer_norm(s_d, weight=blk.pre_norm_s_weight,
                             bias=blk.pre_norm_s_bias, epsilon=1e-5,
                             compute_kernel_config=ckc)
    d_attn_d = blk.attention_pair_bias(s_norm, z_d, seq_mask=None)
    ttnn.deallocate(s_norm)
    d_attn_h = ttnn.to_torch(d_attn_d).float().reshape(n_token, -1)
    s1_d = ttnn.add_(s_d, d_attn_d)
    ttnn.deallocate(d_attn_d)
    d_trans_d = blk.transition_s(s1_d)
    d_trans_h = ttnn.to_torch(d_trans_d).float().reshape(n_token, -1)
    s2_d = ttnn.add_(s1_d, d_trans_d)
    s2_h = ttnn.to_torch(s2_d).float().reshape(n_token, -1)

    s1_ref = s_b[0] + d_attn_ref[0]
    s2_ref = s_b[0] + d_attn_ref[0] + d_trans_ref[0]
    print(f"delta attn_pair_bias:    pcc={pcc(d_attn_h, d_attn_ref[0]):.6f} "
          f"(dev std {float(d_attn_h.std()):.2f} vs ref {float(d_attn_ref[0].std()):.2f})")
    print(f"delta transition_s:      pcc={pcc(d_trans_h, d_trans_ref[0]):.6f} "
          f"(dev std {float(d_trans_h.std()):.2f} vs ref {float(d_trans_ref[0].std()):.2f})")
    print(f"block s out:             pcc={pcc(s2_h, s2_ref):.6f}")


if __name__ == "__main__":
    main()
