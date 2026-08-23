"""Build tt-bio's `Protenix` from every PXDesign-pinned Protenix checkpoint.

The depth and width derivations are only worth what the whole model tree is worth, and a
wrong count shows up as a shape error three modules down, not at the constructor. This walks
all four release checkpoints plus tt-bio's own protenix-v2 and reports, per checkpoint, the
derived tree and whether every stage builds -- input embedder, trunk, diffusion module and
confidence head. PXDesign's generator has no trunk and no confidence head, so `Protenix` cannot build it --
for that one the probe instead checks the claim the port rests on, that its
`input_embedder.atom_attention_encoder` subtree is key-for-key and shape-for-shape the one
tt-bio already runs for protenix-v2, modulo the `design_condition_embedder.` prefix.

Usage:  TT_VISIBLE_DEVICES=1 python3 scripts/pxdesign_port/build_variants_probe.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch

CKPTS = [
    ("protenix-v2 (tt-bio)", os.path.expanduser("~/protenix_ckpt/protenix-v2.pt")),
    ("protenix_base_default_v0.5.0", os.path.expanduser(
        "~/pxdesign_release_data/checkpoint/protenix_base_default_v0.5.0.pt")),
    ("protenix_mini_tmpl_v0.5.0", os.path.expanduser(
        "~/pxdesign_release_data/checkpoint/protenix_mini_tmpl_v0.5.0.pt")),
    ("protenix_mini_default_v0.5.0", os.path.expanduser(
        "~/pxdesign_release_data/checkpoint/protenix_mini_default_v0.5.0.pt")),
    ("pxdesign_v0.1.0 (generator)", os.path.expanduser(
        "~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt")),
]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def main():
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.protenix import Protenix, Trunk, n_blocks

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    rows = []
    for name, path in CKPTS:
        if not os.path.exists(path):
            rows.append(dict(name=name, error="checkpoint missing"))
            print(json.dumps(rows[-1]), flush=True)
            continue
        sd = load(path)
        dm = {k[len("diffusion_module."):]: v for k, v in sd.items()
              if k.startswith("diffusion_module.")}
        tree = dict(
            c_z=Trunk._derive_c_z(sd),
            pairformer=n_blocks(sd, "pairformer_stack"),
            msa=n_blocks(sd, "msa_module"),
            template=n_blocks(sd, "template_embedder.pairformer_stack"),
            confidence=n_blocks(sd, "confidence_head.pairformer_stack"),
            dit=n_blocks(dm, "diffusion_transformer"),
            atom_enc=n_blocks(dm, "atom_attention_encoder.atom_transformer.diffusion_transformer"),
            atom_dec=n_blocks(dm, "atom_attention_decoder.atom_transformer.diffusion_transformer"),
        )
        err = None
        if not tree["pairformer"]:                 # the generator: no trunk, no confidence head
            v2 = load(CKPTS[0][1]) if os.path.exists(CKPTS[0][1]) else None
            gp = "design_condition_embedder.input_embedder.atom_attention_encoder."
            mine = {k[len(gp):]: tuple(v.shape) for k, v in sd.items() if k.startswith(gp)}
            theirs = ({k[len("input_embedder.atom_attention_encoder."):]: tuple(v.shape)
                       for k, v in v2.items()
                       if k.startswith("input_embedder.atom_attention_encoder.")} if v2 else None)
            rows.append(dict(name=name, derived=tree, built=None,
                             atom_encoder_keys=len(mine),
                             atom_encoder_matches_v2=(mine == theirs) if theirs else None,
                             error="no trunk / no confidence head: needs the ProtenixDesign class"))
            print(json.dumps(rows[-1]), flush=True)
            del sd
            continue
        try:
            model = Protenix(sd, ckc, device=dev)
            built = dict(trunk_c_z=model.trunk.C_Z, pairformer=len(model.trunk.PF.blocks),
                         msa=len(model.trunk.MSA), template=len(model.trunk.TPL))
            del model
        except Exception as e:
            built, err = None, f"{type(e).__name__}: {e}"[:300]
        rows.append(dict(name=name, derived=tree, built=built, error=err))
        print(json.dumps(rows[-1]), flush=True)
        del sd

    out = os.path.join(os.path.dirname(__file__), "build_variants_probe.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
