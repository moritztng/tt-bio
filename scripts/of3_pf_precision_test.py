"""Precision discriminator for the 7XI5 pairformer s-track collapse: run the
REFERENCE pairformer stack (CPU torch) in bf16 vs fp32 on the cycle-2 inputs captured
from the reference trunk. If bf16 collapses the same way the device did
(s pcc ~0.77), the defect is intrinsic bf16 ill-conditioning of the s-track for this
input; if bf16 stays clean, the defect is device-side logic.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref /home/ttuser/tt-bio-dev/env/bin/python3 \
    scripts/of3_pf_precision_test.py --query-json /tmp/p13_query_7XI5.json
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

    ref_cap = {"s_glue": [], "z_msa": [], "s_pf": [], "z_pf": []}

    def pf_pre(_m, _args, kwargs):
        ref_cap["s_glue"].append(kwargs["s"].detach().clone())
        ref_cap["z_msa"].append(kwargs["z"].detach().clone())

    def pf_post(_m, _args, out):
        s, z = out
        ref_cap["s_pf"].append(s.detach().clone())
        ref_cap["z_pf"].append(z.detach().clone())

    ref_model.pairformer_stack.register_forward_pre_hook(pf_pre, with_kwargs=True)
    ref_model.pairformer_stack.register_forward_hook(pf_post)

    with torch.no_grad():
        ref_model.run_trunk(batch, num_cycles=4)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]

    for cyc in range(4):
        s_in, z_in = ref_cap["s_glue"][cyc], ref_cap["z_msa"][cyc]
        s_ref, z_ref = ref_cap["s_pf"][cyc], ref_cap["z_pf"][cyc]
        # bf16 emulation: cast stack inputs; module weights are fp32, so temporarily
        # run with autocast bf16 to emulate the device's bf16 activations.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            s_bf, z_bf = ref_model.pairformer_stack(
                s=s_in.clone(), z=z_in.clone(),
                single_mask=token_mask.to(dtype=z_in.dtype),
                pair_mask=pair_mask.to(dtype=s_in.dtype),
                chunk_size=None, use_deepspeed_evo_attention=False,
                use_triton_triangle_kernels=False, use_cueq_triangle_kernels=False,
                use_lma=False, inplace_safe=False, _mask_trans=True,
            )
        print(f"cycle {cyc}: pf-in s std {float(s_in.std()):.1f} | "
              f"fp32-out s std {float(s_ref.std()):.1f} | "
              f"bf16-emul vs fp32: s pcc={pcc(s_bf.float(), s_ref.float()):.5f} "
              f"z pcc={pcc(z_bf.float(), z_ref.float()):.5f}")


if __name__ == "__main__":
    main()
