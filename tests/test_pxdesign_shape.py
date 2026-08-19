"""Architecture depths must come from the checkpoint, not from a constant.

tt_bio.protenix used to hardcode four block counts (DiT 24, atom encoder/decoder 3,
MSA 4, template embedder 2). Those are right for the two v2-family checkpoints tt-bio
shipped -- protenix-v2 and OpenDDE -- and wrong for every Protenix variant PXDesign pins:
PXDesign's generator is 16 DiT / 4+4 atom, the Protenix mini variants are 8 / 1+1, and all
three have a template embedder with no pairformer stack at all.

The two guarantees this test exists to hold:
  1. `n_blocks` reproduces the OLD hardcoded numbers for protenix-v2, so the derivation is
     a no-op for the shipped model. If this arm fails, the derivation moved production.
  2. It reports the PXDesign numbers, so a regression back to a constant is caught here
     rather than as a silently wrong fold.

The same argument applies to the pair WIDTH: c_z is 256 for protenix-v2, 384 for OpenDDE and
128 for every PXDesign-pinned Protenix, and `Trunk` used to need it threaded in by hand.
`Trunk._derive_c_z` reads it off `layernorm_z_cycle`, and the arms below pin that it returns
the shipped models' existing widths.

Device-free: reads key names and tensor shapes only, no ttnn, no forward pass.
"""
import os

import pytest
import torch

from tt_bio.protenix import Trunk, n_blocks

_CKPT_DIR = os.path.expanduser("~/pxdesign_release_data/checkpoint")
_V2 = os.path.expanduser("~/protenix_ckpt/protenix-v2.pt")

# (file, pairformer, msa, template, confidence_pairformer, dit, atom_enc, atom_dec)
_EXPECTED = [
    ("pxdesign_v0.1.0.pt", 0, 0, 0, 0, 16, 4, 4),
    ("protenix_base_default_v0.5.0.pt", 48, 4, 0, 4, 24, 3, 3),
    ("protenix_mini_tmpl_v0.5.0.pt", 16, 1, 0, 4, 8, 1, 1),
    ("protenix_mini_default_v0.5.0.pt", 16, 1, 0, 4, 8, 1, 1),
]


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def _diffusion(sd):
    return {k[len("diffusion_module."):]: v for k, v in sd.items() if k.startswith("diffusion_module.")}


def test_n_blocks_absent_stack_is_zero():
    """A missing stack is 0, not a crash and not a 1."""
    assert n_blocks({}, "diffusion_transformer") == 0
    assert n_blocks({"unrelated.weight": None}, "diffusion_transformer") == 0


def test_n_blocks_anchors_the_prefix():
    """`pairformer_stack` must not count `template_embedder.pairformer_stack` or
    `confidence_head.pairformer_stack`. The previous inline derivation used an unanchored
    search over any key containing 'pairformer_stack.blocks.' and took the max index, which
    happened to be right only because the trunk stack is the deepest one in v2."""
    sd = {
        "pairformer_stack.blocks.0.x": None,
        "pairformer_stack.blocks.1.x": None,
        "confidence_head.pairformer_stack.blocks.7.x": None,
        "template_embedder.pairformer_stack.blocks.5.x": None,
    }
    assert n_blocks(sd, "pairformer_stack") == 2
    assert n_blocks(sd, "confidence_head.pairformer_stack") == 8
    assert n_blocks(sd, "template_embedder.pairformer_stack") == 6


@pytest.mark.skipif(not os.path.exists(_V2), reason="needs ~/protenix_ckpt/protenix-v2.pt")
def test_v2_derivation_reproduces_the_old_hardcoded_depths():
    """The no-op guarantee: protenix-v2 must derive exactly what was hardcoded before."""
    sd = _load(_V2)
    dm = _diffusion(sd)
    assert n_blocks(dm, "diffusion_transformer") == 24                      # was DIT_BLOCKS = 24
    assert n_blocks(dm, "atom_attention_encoder.atom_transformer.diffusion_transformer") == 3
    assert n_blocks(dm, "atom_attention_decoder.atom_transformer.diffusion_transformer") == 3
    assert n_blocks(sd, "msa_module") == 4                                  # was nb_msa = 4
    assert n_blocks(sd, "template_embedder.pairformer_stack") == 2          # was range(2)
    assert n_blocks(sd, "pairformer_stack") == 48


@pytest.mark.parametrize("case", _EXPECTED, ids=[c[0].replace(".pt", "") for c in _EXPECTED])
def test_pxdesign_family_depths(case):
    name, pf, msa, tpl, conf, dit, enc, dec = case
    path = os.path.join(_CKPT_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} missing; fetch via scripts/pxdesign_port/fetch_release_data.sh")
    sd = _load(path)
    dm = _diffusion(sd)
    assert n_blocks(sd, "pairformer_stack") == pf
    assert n_blocks(sd, "msa_module") == msa
    assert n_blocks(sd, "template_embedder.pairformer_stack") == tpl
    assert n_blocks(sd, "confidence_head.pairformer_stack") == conf
    assert n_blocks(dm, "diffusion_transformer") == dit
    assert n_blocks(dm, "atom_attention_encoder.atom_transformer.diffusion_transformer") == enc
    assert n_blocks(dm, "atom_attention_decoder.atom_transformer.diffusion_transformer") == dec


def test_c_z_derivation_is_none_without_trunk_keys():
    """No trunk in the dict (PXDesign's generator) -> None, so the class default still applies."""
    assert Trunk._derive_c_z({}) is None
    assert Trunk.C_Z == 256


def test_c_z_derivation_reads_the_norm_it_is_given():
    """384 is OpenDDE's width. No OpenDDE checkpoint lives on the gate hosts, so the arm is
    synthetic -- what it pins is that the derivation is the norm's length and nothing else."""
    assert Trunk._derive_c_z({"layernorm_z_cycle.weight": torch.zeros(384)}) == 384
    assert Trunk._derive_c_z({"layernorm_z_cycle.weight": torch.zeros(256)}) == 256


@pytest.mark.skipif(not os.path.exists(_V2), reason="needs ~/protenix_ckpt/protenix-v2.pt")
def test_c_z_derivation_reproduces_the_v2_default():
    """The no-op guarantee for the width, matching the one above for the depths."""
    assert Trunk._derive_c_z(_load(_V2)) == Trunk.C_Z == 256


@pytest.mark.parametrize("name", [c[0] for c in _EXPECTED[1:]])
def test_pxdesign_protenix_variants_are_c_z_128(name):
    path = os.path.join(_CKPT_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} missing; fetch via scripts/pxdesign_port/fetch_release_data.sh")
    assert Trunk._derive_c_z(_load(path)) == 128
