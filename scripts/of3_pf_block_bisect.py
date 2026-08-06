"""7XI5 s-track block bisect: reference pairformer per-block trajectory vs the device
Pairformer run block-by-block on the SAME cycle-2 inputs (captured from the reference
trunk). Finds the block where the device s-track collapses.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_pf_block_bisect.py \
    --query-json /tmp/p13_query_7XI5.json [--cycle 2]
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

    # Reference per-block trajectory on the captured inputs.
    n_blocks = len(ref_model.pairformer_stack.blocks)
    ref_s, ref_z = [], []
    handles = []

    def mk_post(_):
        def hook(_m, _args, out):
            s, z = out
            ref_s.append(s.detach().clone())
            ref_z.append(z.detach().clone())
        return hook

    for b in ref_model.pairformer_stack.blocks:
        handles.append(b.register_forward_hook(mk_post(b)))
    with torch.no_grad():
        ref_model.pairformer_stack(
            s=s_in.clone(), z=z_in.clone(),
            single_mask=token_mask.to(dtype=z_in.dtype),
            pair_mask=pair_mask.to(dtype=s_in.dtype),
            chunk_size=None, use_deepspeed_evo_attention=False,
            use_triton_triangle_kernels=False, use_cueq_triangle_kernels=False,
            use_lma=False, inplace_safe=False, _mask_trans=True,
        )
    for h in handles:
        h.remove()
    print(f"reference: {len(ref_s)} blocks captured at cycle {args.cycle}")

    # Device per-block trajectory on the same inputs.
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

    trunk = OF3Trunk(sd, ckc, num_cycles=1)  # only .pairformer is used
    n_token = int(s_in.shape[1])
    s_d, z_d = ft(s_in), ft(z_in)
    print(f"{'blk':>4} {'s_pcc':>9} {'z_pcc':>9} {'s_std_dev':>10} {'s_std_ref':>10}")
    for i, block in enumerate(trunk.pairformer.blocks):
        s_d, z_d = block(s_d, z_d, None, None, None, None)
        s_h = ttnn.to_torch(s_d).float().reshape(n_token, -1)
        z_h = ttnn.to_torch(z_d).float().reshape(n_token, n_token, -1)
        if i % 4 == 3 or i == 0 or i == n_blocks - 1:
            print(f"{i:>4} {pcc(s_h, ref_s[i][0]):>9.5f} {pcc(z_h, ref_z[i][0]):>9.5f} "
                  f"{float(s_h.std()):>10.1f} {float(ref_s[i][0].std()):>10.1f}")
    # find the knee: worst drop between consecutive blocks
    pccs = []
    s_d2, z_d2 = ft(s_in), ft(z_in)
    for i, block in enumerate(trunk.pairformer.blocks):
        s_d2, z_d2 = block(s_d2, z_d2, None, None, None, None)
        s_h = ttnn.to_torch(s_d2).float().reshape(n_token, -1)
        pccs.append(pcc(s_h, ref_s[i][0]))
    drops = [(pccs[i - 1] - pccs[i], i) for i in range(1, len(pccs))]
    drops.sort(reverse=True)
    print("largest s_pcc drops (delta, block):", [(round(d, 4), b) for d, b in drops[:5]])


if __name__ == "__main__":
    main()
