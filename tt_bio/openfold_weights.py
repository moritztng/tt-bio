"""OpenFold reference-checkpoint -> tt_bio.openfold weight remaps.

Scopes a real (aqlaboratory/openfold, non-fused-projection) AlphaFold checkpoint's
Evoformer trunk into the per-sub-block state_dicts that tt_bio.openfold.EvoformerBlock /
EvoformerStack consume. Pure dict/tensor ops (no model import). Standard AF2 weights use
the non-fused TriangleMultiplication layout (layer_norm_in / linear_a_p / ...), which
protenix_weights.remap_triangle_multiplication targets directly.
"""
from __future__ import annotations

from tt_bio.protenix_weights import remap_triangle_multiplication, remap_outer_product_mean


def _scope(sd: dict, *prefixes: str) -> dict:
    """Scope by the first prefix that matches any key. The pair-track / transition /
    OPM ops live under `core.` in the released OpenFold checkpoints (finetuning_*.pt)
    but under `pair_stack.` / directly on the block in the current vendored reference —
    accept both so one loader handles either layout."""
    for p in prefixes:
        out = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        if out:
            return out
    return {}


def _strip_mha(sd: dict) -> dict:
    return {k.replace("mha.", ""): v for k, v in sd.items()}


def evoformer_block_subs(block_sd: dict) -> dict:
    """One reference/checkpoint EvoformerBlock state_dict -> the sub-block dicts
    EvoformerBlock wants. Layout-robust (`core.` released-ckpt vs `pair_stack.`/direct
    vendored)."""
    return {
        "row": _scope(block_sd, "msa_att_row."),
        "col": _scope(block_sd, "msa_att_col."),           # keeps _msa_att.; block strips it
        "msa_transition": _scope(block_sd, "core.msa_transition.", "msa_transition."),
        "opm": remap_outer_product_mean(_scope(block_sd, "core.outer_product_mean.", "outer_product_mean.")),
        "tri_mul_out": remap_triangle_multiplication(_scope(block_sd, "core.tri_mul_out.", "pair_stack.tri_mul_out.")),
        "tri_mul_in": remap_triangle_multiplication(_scope(block_sd, "core.tri_mul_in.", "pair_stack.tri_mul_in.")),
        "tri_att_start": _strip_mha(_scope(block_sd, "core.tri_att_start.", "pair_stack.tri_att_start.")),
        "tri_att_end": _strip_mha(_scope(block_sd, "core.tri_att_end.", "pair_stack.tri_att_end.")),
        "pair_transition": _scope(block_sd, "core.pair_transition.", "pair_stack.pair_transition."),
    }


def evoformer_stack_subs(stack_sd: dict, no_blocks: int):
    """Reference EvoformerStack state_dict -> (per-block sub-dicts, s-projection dict)."""
    subs = [evoformer_block_subs(_scope(stack_sd, f"blocks.{i}.")) for i in range(no_blocks)]
    return subs, _scope(stack_sd, "linear.")
