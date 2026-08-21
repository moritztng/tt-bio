"""Census the architecture shape of a Protenix-family checkpoint from its keys alone.

    python3 scripts/pxdesign_port/ckpt_shape.py <ckpt.pt> [<ckpt.pt> ...]

The block counts come from tt_bio.protenix.n_blocks, the same function the port itself uses,
so this reports what the model will actually build rather than a second opinion. Handy when a
new family variant shows up: PXDesign pins three Protenix variants at depths tt-bio had
hardcoded away (see tests/test_pxdesign_shape.py for the numbers this locks in).
"""

import os
import sys

import torch

from tt_bio.protenix import n_blocks

PAIR_PROBE = "pairformer_stack.blocks.0.tri_mul_out.linear_z.weight"


def shape(model_state_dict):
    """Block counts + pair width for a v2-family model dict (`module.` prefix STRIPPED)."""
    sd = model_state_dict
    dm = {k[len("diffusion_module."):]: v for k, v in sd.items() if k.startswith("diffusion_module.")}
    out = {
        "pairformer": n_blocks(sd, "pairformer_stack"),
        "msa": n_blocks(sd, "msa_module"),
        "template": n_blocks(sd, "template_embedder.pairformer_stack"),
        "confidence_pairformer": n_blocks(sd, "confidence_head.pairformer_stack"),
        # c_z is read off a pair-shaped weight, not assumed: protenix-v2 is 256, OpenDDE 384,
        # every PXDesign-pinned Protenix variant 128.
        "c_z": int(sd[PAIR_PROBE].shape[0]) if PAIR_PROBE in sd else None,
    }
    if dm:
        out.update({
            "dit": n_blocks(dm, "diffusion_transformer"),
            "atom_encoder": n_blocks(dm, "atom_attention_encoder.atom_transformer.diffusion_transformer"),
            "atom_decoder": n_blocks(dm, "atom_attention_decoder.atom_transformer.diffusion_transformer"),
        })
    return out


def main(paths):
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        print(os.path.basename(p), f"  {len(sd)} tensors  {n / 1e6:.2f} M params")
        for k, v in shape(sd).items():
            print(f"    {k:22s} {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
