"""Full-bf16 emulation of the pairformer s-track for the 7XI5 knife-edge: casts the
ENTIRE reference pairformer stack (weights, LayerNorm, softmax, residuals) to bf16 —
matching the device's regime — vs autocast-bf16 (which keeps LN/softmax fp32) and
fp32, all on the same captured cycle inputs.

Outcome reading:
  - full-bf16 collapses (s pcc ~0.8) while autocast stays 1.0 -> the device is
    faithfully reproducing a bf16-intrinsic knife-edge; the reference survives via
    fp32 LN/softmax; fix = selective-fp32 s-track boundary (precedent:
    af3-diffusion-sampler-selective-fp32-boundary).
  - full-bf16 also stays clean -> the device has a genuine logic/precision defect
    beyond dtype; bisect the device AttentionPairBias op next.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref /home/ttuser/tt-bio-dev/env/bin/python3 \
    scripts/of3_pf_fullbf16_test.py --query-json /tmp/p13_query_7XI5.json
"""
import argparse
import copy
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

    cap = {"s_glue": [], "z_msa": [], "s_pf": [], "z_pf": []}

    def pf_pre(_m, _args, kwargs):
        cap["s_glue"].append(kwargs["s"].detach().clone())
        cap["z_msa"].append(kwargs["z"].detach().clone())

    def pf_post(_m, _args, out):
        s, z = out
        cap["s_pf"].append(s.detach().clone())
        cap["z_pf"].append(z.detach().clone())

    ref_model.pairformer_stack.register_forward_pre_hook(pf_pre, with_kwargs=True)
    ref_model.pairformer_stack.register_forward_hook(pf_post)
    with torch.no_grad():
        ref_model.run_trunk(batch, num_cycles=4)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    pf_bf16 = copy.deepcopy(ref_model.pairformer_stack).bfloat16().eval()

    for cyc in (0, 2, 3):
        s_in, z_in = cap["s_glue"][cyc], cap["z_msa"][cyc]
        s_ref = cap["s_pf"][cyc]
        with torch.no_grad():
            s_bf, z_bf = pf_bf16(
                s=s_in.bfloat16(), z=z_in.bfloat16(),
                single_mask=token_mask.to(torch.bfloat16),
                pair_mask=pair_mask.to(torch.bfloat16),
                chunk_size=None, use_deepspeed_evo_attention=False,
                use_triton_triangle_kernels=False, use_cueq_triangle_kernels=False,
                use_lma=False, inplace_safe=False, _mask_trans=True,
            )
        print(f"cycle {cyc}: FULL-bf16 reference stack vs fp32: "
              f"s pcc={pcc(s_bf.float(), s_ref.float()):.5f} "
              f"(s std in {float(s_in.std()):.1f} out {float(s_ref.std()):.1f})")


if __name__ == "__main__":
    main()
