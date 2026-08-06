"""A/B inside the device AttentionPairBias for the 7XI5 knife-edge (cycle 2, block 44):

  1. Runs the device op front (qkv linear -> heads; bias via compute_bias) and pulls
     q/k/v/bias to host; PCCs them against the reference mha/_prep_bias internals.
  2. Variant FUSED: ttnn.transformer.scaled_dot_product_attention (the shipping path).
  3. Variant MANUAL: ttnn matmul + ttnn.softmax + matmul with the same q/k/v/bias.
  Both continue through the device gate + out-projection; each delta is PCC/std-ratio'd
  against the reference block's attn delta. If MANUAL is clean and FUSED is not, the
  fused SDPA kernel's internal softmax is the defect.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_apb_sdpa_ab.py \
    --query-json /tmp/p13_query_7XI5.json [--cycle 2] [--block 44]
"""
import argparse
import math
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

    # reference q/k/v/bias from the reference mha internals (fp64 host)
    apb = blk_ref.attn_pair_bias
    a_ref = apb.layer_norm_a(s_b[0])
    with torch.no_grad():
        q_ref, k_ref, v_ref = apb.mha._prep_qkv(a_ref, a_ref, apply_scale=True)
        biases = apb._prep_bias(a=a_ref, z=z_mid_ref, mask=None)
        bias_ref = sum(biases)  # [H, N, N] (mask_bias is all zeros)
    print(f"reference internals: q {tuple(q_ref.shape)} bias {tuple(bias_ref.shape)}")

    # ---- device ------------------------------------------------------------
    import ttnn

    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    def ft(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    trunk = OF3Trunk(sd, ckc, num_cycles=1)
    op = trunk.pairformer.blocks[B].attention_pair_bias
    n_token = int(s_in.shape[1])
    H, D = op.n_heads, op.head_dim

    s_dev = ft(a_ref.unsqueeze(0))
    z_dev = ft(z_mid_ref.unsqueeze(0))

    qkv = ttnn.linear(s_dev, op.qkv_weight, bias=op.qkv_bias,
                      compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
    qkv = ttnn.unsqueeze(qkv, 1)
    q_d, k_d, v_d = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
    ttnn.deallocate(qkv)
    bias_d = op.compute_bias(z_dev)  # (1, H, S, S), sqrt(D)-premultiplied

    q_h = ttnn.to_torch(q_d).float()[0, :, :n_token, :D]
    k_h = ttnn.to_torch(k_d).float()[0, :, :n_token, :D]
    v_h = ttnn.to_torch(v_d).float()[0, :, :n_token, :D]
    bias_h = ttnn.to_torch(bias_d).float()[0, :, :n_token, :n_token]
    print(f"device front vs reference: q/sqrtD pcc={pcc(q_h, q_ref / math.sqrt(D)):.5f} "
          f"raw-q pcc={pcc(q_h, q_ref):.5f}  "
          f"k pcc={pcc(k_h, k_ref):.5f} v pcc={pcc(v_h, v_ref):.5f} "
          f"bias pcc={pcc(bias_h, bias_ref):.5f} bias*sqrtD pcc={pcc(bias_h, bias_ref * math.sqrt(D)):.5f}")

    def finish(o_dev):
        o = o_dev[:, :, :, :D]
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
        o = ttnn.permute(o, (0, 2, 1))
        g = ttnn.linear(s_dev, op.g_weight, compute_kernel_config=ckc,
                        core_grid=CORE_GRID_MAIN)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                          dtype=ttnn.bfloat16)
        ttnn.deallocate(g)
        x = ttnn.linear(o, op.o_weight, compute_kernel_config=ckc,
                        core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(o)
        return ttnn.to_torch(x).float().reshape(n_token, -1)

    # variant FUSED (shipping path)
    o_f = ttnn.transformer.scaled_dot_product_attention(
        q_d, k_d, v_d, attn_mask=bias_d, is_causal=False, scale=D**-0.5,
        program_config=None)
    d_fused = finish(o_f)
    print(f"FUSED : pcc={pcc(d_fused, d_attn_ref):.6f} std ratio "
          f"{float(d_fused.std() / d_attn_ref.std()):.4f}")

    # variant MANUAL: logits = (q k^T + bias_sqrtD) / sqrt(D)  [== qk/sqrtD + bias]
    kt = ttnn.transpose(k_d, -2, -1)
    logits = ttnn.matmul(q_d, kt, compute_kernel_config=ckc)
    logits = ttnn.add_(logits, bias_d)
    logits = ttnn.multiply_(logits, D**-0.5)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=ckc)
    ttnn.deallocate(logits)
    o_m = ttnn.matmul(probs, v_d, compute_kernel_config=ckc)
    ttnn.deallocate(probs)
    d_manual = finish(o_m)
    print(f"MANUAL: pcc={pcc(d_manual, d_attn_ref):.6f} std ratio "
          f"{float(d_manual.std() / d_attn_ref.std()):.4f}")

    # host fp64 reference probabilities for a direct softmax comparison
    with torch.no_grad():
        logits_ref = (q_ref @ k_ref.transpose(-1, -2)) + bias_ref
        probs_ref = torch.softmax(logits_ref.double(), dim=-1)
        logits_dev_host = (q_h @ k_h.transpose(-1, -2) + bias_h) / math.sqrt(D)
        probs_dev_host = torch.softmax(logits_dev_host.double(), dim=-1)
        print(f"host-recomputed probs from DEVICE q/k/bias vs reference probs: "
              f"pcc={pcc(probs_dev_host, probs_ref):.6f}")


if __name__ == "__main__":
    main()
