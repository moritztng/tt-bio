"""Remap RF3's atom attention encoder onto tt-bio's shared block names.

Three mappings here are crossed over relative to how the names read, and each was
taken from the reference's forward rather than from the name:

  * `swish_gate.0.weight` is chunk 0 = the NON-SiLU'd operand, chunk 1 = the SiLU'd
    one, so it is cat([linear_2, linear_1]) -- linear_1 is the one RF3 applies SiLU to.
  * RF3 nests AdaLN and the adaLN-Zero output gate INSIDE its attention block, while
    tt-bio hoists both into DiffusionTransformerLayer, so `attention_pair_bias.ada_ln_1.*`
    and `attention_pair_bias.linear_output_project.0.*` lift a level.
  * tt-bio reads `proj_q.bias` unconditionally; RF3's `to_q` is bias-free, so an
    explicit zero bias is supplied rather than letting the load fail.
"""

from __future__ import annotations

import torch

# RF3 AdaLN -> tt-bio AdaLN. Verified against both forwards: tt-bio layer-norms `a`
# with no affine, layer-norms `s` with weight only, and applies sigmoid on s_scale,
# which is exactly `to_gain(ln_s(S)) * ln_a(A) + to_bias(ln_s(S))`.
ADALN = {
    "ln_s.weight": "s_norm.weight",
    "to_gain.0.weight": "s_scale.weight",
    "to_gain.0.bias": "s_scale.bias",
    "to_bias.weight": "s_bias.weight",
}

ATTENTION = {
    "to_q.weight": "proj_q.weight",
    "to_k.weight": "proj_k.weight",
    "to_v.weight": "proj_v.weight",
    "to_g.0.weight": "proj_g.weight",
    "to_a.weight": "proj_o.weight",
    "query_layer_norm.weight": "query_layer_norm.weight",
    "query_layer_norm.bias": "query_layer_norm.bias",
    "key_layer_norm.weight": "key_layer_norm.weight",
    "key_layer_norm.bias": "key_layer_norm.bias",
}


def _adaln(src: dict, prefix: str, dst: dict, out_prefix: str) -> None:
    for a, b in ADALN.items():
        dst[f"{out_prefix}{b}"] = src[f"{prefix}{a}"]


def remap_atom_transformer_block(src: dict, i: int) -> dict:
    """One RF3 DiT block -> tt-bio `layers.{i}.*` names."""
    p = f"blocks.{i}."
    out: dict[str, torch.Tensor] = {}
    o = f"layers.{i}."

    apb = f"{p}attention_pair_bias."
    _adaln(src, f"{apb}ada_ln_1.", out, f"{o}adaln.")
    for a, b in ATTENTION.items():
        out[f"{o}pair_bias_attn.{b}"] = src[f"{apb}{a}"]
    # RF3's to_q carries no bias; tt-bio reads one unconditionally.
    out[f"{o}pair_bias_attn.proj_q.bias"] = torch.zeros(
        src[f"{apb}to_q.weight"].shape[0], dtype=src[f"{apb}to_q.weight"].dtype
    )
    # The adaLN-Zero output gate lives inside RF3's attention, one level up in tt-bio.
    out[f"{o}output_projection_linear.weight"] = src[f"{apb}linear_output_project.0.weight"]
    out[f"{o}output_projection_linear.bias"] = src[f"{apb}linear_output_project.0.bias"]

    ctb = f"{p}conditioned_transition_block."
    _adaln(src, f"{ctb}ada_ln.", out, f"{o}transition.adaln.")
    # chunk 0 is the non-SiLU'd operand: cat([linear_2, linear_1]), NOT the name order.
    out[f"{o}transition.swish_gate.0.weight"] = torch.cat(
        [src[f"{ctb}linear_2.weight"], src[f"{ctb}linear_1.weight"]], dim=0
    )
    out[f"{o}transition.b_to_a.weight"] = src[f"{ctb}linear_3.weight"]
    out[f"{o}transition.output_projection.0.weight"] = src[f"{ctb}linear_output_project.0.weight"]
    out[f"{o}transition.output_projection.0.bias"] = src[f"{ctb}linear_output_project.0.bias"]
    # a_to_b deliberately absent: RF3 is the 2-factor SwiGLU (a_to_b_gate=False).
    return out


def atom_transformer_bias_weights(src: dict, i: int) -> dict:
    """The pair-bias projection for block `i`, which does NOT go into the block.

    tt-bio's `DiffusionTransformerLayer` builds its attention with
    `compute_pair_bias=False` and expects the caller to hand it a ready bias, so
    `ln_0` and `to_b` are consumed here instead. That is not an optimisation: RF3
    masks out-of-range keys additively with `-1e9 * (maskQ + maskK)` on the logits,
    while tt-bio's windowed gather yields ZERO key rows that still enter the softmax.
    The mask has to be folded into the bias, so the bias has to be built outside.
    """
    pre = "atom_transformer.diffusion_transformer."
    p = f"{pre}blocks.{i}.attention_pair_bias."
    return {
        "ln_0.weight": src[f"{p}ln_0.weight"],
        "ln_0.bias": src[f"{p}ln_0.bias"],
        "to_b.weight": src[f"{p}to_b.weight"],
    }


def remap_atom_transformer(src: dict, n_block: int = 3) -> dict:
    """`atom_transformer.diffusion_transformer.*` -> tt-bio DiffusionTransformer names."""
    pre = "atom_transformer.diffusion_transformer."
    inner = {k[len(pre):]: v for k, v in src.items() if k.startswith(pre)}
    out: dict[str, torch.Tensor] = {}
    for i in range(n_block):
        out.update(remap_atom_transformer_block(inner, i))
    return out


def atom_transformer_unmapped(src: dict, n_block: int = 3) -> list[str]:
    """RF3 keys under the atom transformer that the remap does not consume.

    The pre-flight equivalent for weights. It cannot see weightless config flags --
    `no_residual`, `use_inv_dist_squared`, `a_to_b_gate` were all missed that way --
    so it is necessary and not sufficient.
    """
    pre = "atom_transformer.diffusion_transformer."
    inner = {k[len(pre):] for k in src if k.startswith(pre)}
    consumed = set()
    for i in range(n_block):
        p = f"blocks.{i}."
        apb, ctb = f"{p}attention_pair_bias.", f"{p}conditioned_transition_block."
        consumed |= {f"{apb}{a}" for a in ATTENTION}
        consumed |= {f"{apb}ada_ln_1.{a}" for a in ADALN}
        consumed |= {f"{ctb}ada_ln.{a}" for a in ADALN}
        # consumed by atom_transformer_bias_weights(), not by the block itself
        consumed |= {f"{apb}ln_0.weight", f"{apb}ln_0.bias", f"{apb}to_b.weight"}
        consumed |= {
            f"{apb}linear_output_project.0.weight", f"{apb}linear_output_project.0.bias",
            f"{ctb}linear_1.weight", f"{ctb}linear_2.weight", f"{ctb}linear_3.weight",
            f"{ctb}linear_output_project.0.weight", f"{ctb}linear_output_project.0.bias",
        }
    return sorted(inner - consumed)
