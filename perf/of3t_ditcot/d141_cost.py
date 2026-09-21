#!/usr/bin/env python3
"""D141 priced: what does the reference's all-ones layer_norm_z actually cost?

The probe settled that the diffusion reference runs one shared `layer_norm_z` at its all-ones
init while our port runs the checkpoint's 48 trained per-block tensors. That is not a loading
bug: upstream 0.5.0's `DiffusionAttentionPairBias` has no `layer_norm_z` member at all, so the
package cannot represent the architecture the checkpoint was trained in and no strict=True or
extra flag would recover it.

Our port CAN run both layouts -- `OF3DiffusionTransformer` picks on the state dict, taking the
shared pre-stack norm when `layer_norm_z.weight` is present and per-block norms when it is not
(`tt_bio/openfold3_diffusion_transformer.py:268-274`). So the cost of the architecture
difference is measurable on the one side that can do both:

  ARM ours      the checkpoint as trained -- 24 per-block trained layer_norm_z
  ARM refarch   the reference's architecture -- the per-block tensors deleted, one shared
                layer_norm_z at all-ones, exactly what load_state_dict(strict=False) leaves

Both run on the card, on the same boundary, same cotangent, same everything else. The gap
between them is what the denominator in every of3t ratio has been absorbing.

Measurement only. Nothing in tt_bio/ is edited; the state dict is rewritten in this process.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_diffusion"))

C_Z = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lnz-arm", required=True, dest="lnz_arm",
                    choices=("ours", "refarch", "refarch_rand"))
    a, rest = ap.parse_known_args()

    import device_gradient as DG

    real_load = torch.load
    tag = a.lnz_arm

    def patched(path, *pa, **pk):
        sd = real_load(path, *pa, **pk)
        if not tag.startswith("refarch") or not isinstance(sd, dict):
            return sd
        inner = sd.get("state_dict", sd)
        if not any("diffusion_transformer" in k for k in inner):
            return sd
        # Drop every per-block layer_norm_z and install one shared all-ones norm, which is
        # precisely the state the reference builder is left in.
        drop = [k for k in inner
                if "diffusion_transformer.blocks." in k
                and "attention_pair_bias.layer_norm_z" in k]
        for k in drop:
            del inner[k]
        ref = next(iter(inner.values()))
        for pre in ("diffusion_module.", "sample_diffusion.diffusion_module."):
            key = pre + "diffusion_transformer.layer_norm_z.weight"
            dt = ref.dtype if ref.is_floating_point() else torch.float32
            if tag == "refarch_rand":
                # CONTROL. The shared LAYOUT with a weight that is not the all-ones init, so a
                # gain measured for refarch can be attributed to matching the reference rather
                # than to the layout being kinder on its own.
                gen = torch.Generator().manual_seed(20260921)
                inner[key] = (torch.randn(C_Z, generator=gen) * 0.2 + 1.0).to(dt)
            else:
                inner[key] = torch.ones(C_Z, dtype=dt)
        what = "random N(1, 0.2)" if tag == "refarch_rand" else "all-ones"
        print(f"[{tag}] dropped {len(drop)} per-block layer_norm_z, installed "
              f"shared {what}", flush=True)
        return sd

    torch.load = patched
    sys.argv = [sys.argv[0]] + rest
    return DG.main()


if __name__ == "__main__":
    raise SystemExit(main())
