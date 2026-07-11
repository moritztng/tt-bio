"""OpenFold reference-checkpoint -> tt_bio.openfold weight remaps.

Scopes a real (aqlaboratory/openfold, non-fused-projection) AlphaFold checkpoint's
Evoformer trunk into the per-sub-block state_dicts that tt_bio.openfold.EvoformerBlock /
EvoformerStack consume. Pure dict/tensor ops (no model import). Standard AF2 weights use
the non-fused TriangleMultiplication layout (layer_norm_in / linear_a_p / ...), which
protenix_weights.remap_triangle_multiplication targets directly.
"""
from __future__ import annotations

from tt_bio.protenix_weights import remap_triangle_multiplication, remap_outer_product_mean


def _scope(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _strip_mha(sd: dict) -> dict:
    return {k.replace("mha.", ""): v for k, v in sd.items()}


def evoformer_block_subs(block_sd: dict) -> dict:
    """One reference EvoformerBlock state_dict -> the sub-block dicts EvoformerBlock wants."""
    return {
        "row": _scope(block_sd, "msa_att_row."),
        "col": _scope(block_sd, "msa_att_col."),           # keeps _msa_att.; block strips it
        "msa_transition": _scope(block_sd, "msa_transition."),
        "opm": remap_outer_product_mean(_scope(block_sd, "outer_product_mean.")),
        "tri_mul_out": remap_triangle_multiplication(_scope(block_sd, "pair_stack.tri_mul_out.")),
        "tri_mul_in": remap_triangle_multiplication(_scope(block_sd, "pair_stack.tri_mul_in.")),
        "tri_att_start": _strip_mha(_scope(block_sd, "pair_stack.tri_att_start.")),
        "tri_att_end": _strip_mha(_scope(block_sd, "pair_stack.tri_att_end.")),
        "pair_transition": _scope(block_sd, "pair_stack.pair_transition."),
    }


def evoformer_stack_subs(stack_sd: dict, no_blocks: int):
    """Reference EvoformerStack state_dict -> (per-block sub-dicts, s-projection dict)."""
    subs = [evoformer_block_subs(_scope(stack_sd, f"blocks.{i}.")) for i in range(no_blocks)]
    return subs, _scope(stack_sd, "linear.")
