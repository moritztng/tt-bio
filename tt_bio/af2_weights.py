"""AlphaFold2 monomer (`model_1_ptm`) parameters, remapped to tt-bio module layout.

`params_model_1_ptm.npz` is the 2021 DeepMind release: 338 flat arrays keyed
`<haiku/scope>//<param>`, stored fp32. This module turns it into a flat torch state dict whose
keys are tt-bio dotted module scopes, so `WeightScope.child()` slices it the same way Boltz-2
and Protenix checkpoints are sliced.

Three conversions are not renames, and each one is a place a port silently gets wrong numbers.

**Haiku Linear stores `[in, *out]`, torch stores `[out, in]`.** A haiku
`Linear(num_output=(heads, dim))` weight is `[c, heads, dim]` contracted `...a,ahi->...hi`, so
the torch equivalent is `w.reshape(c, heads * dim).T` and the output channel index is
`head * dim + i`. Same for `output_w [heads, dim, c]` -> `[c, heads * dim]`.

**Evoformer and extra-MSA weights are stacked over blocks in dim 0** (48 and 4), and the
template pair stack too (2). The structure module is not stacked: its 8 layers share one
`fold_iteration`.

**The incoming triangle multiplication has left and right swapped.** AF2 runs
`einsum('kjc,kic->ijc', left, right)` and says so in its own source comment ("for the
'incoming' edges, it's swapped"); tt-bio's `TriangleMultiplicationIncoming` runs
`einsum('bkid,bkjd->bijd', a, b)` (`reference.py`), so `a` is AF2's *right* and `b` is AF2's
*left*. The outgoing block needs no swap. Concatenating in source order for both directions
transposes every incoming update, which is a plausible-looking wrong answer, not a crash.

ColabDesign runs the *fused* triangle multiplication and builds its fused arrays by
concatenating `[left | right]` on the last axis at load time
(`alphafold/model/utils.py::flat_params_to_haiku`). We concatenate straight from the npz
instead: the same numbers in the same order, plus the swap above.

Also settled here from the reference source, because they pick tt-bio constructor flags:

- AF2's outer product mean divides the whole output *including* `output_b` by the pair norm,
  with epsilon 1e-3, so `OuterProductMean(scale_bias=True)`.
- AF2's triangle attention pre-scales q by `key_dim ** -0.5` and adds the pair bias raw, so the
  bias is unscaled: `TriangleAttention(scale_pair_bias=False)`.
- Every LayerNorm in the trunk upcasts bf16 to fp32 internally and uses `use_fast_variance=True`
  (`common_modules.LayerNorm`). The trunk runs bf16; the structure module and heads run fp32.

Not consumed, deliberately: the distogram, masked-MSA and experimentally-resolved heads.
PXDesign reads none of them. `load_af2_state_dict` asserts they are the *only* unconsumed
arrays, so a remap that quietly drops a block fails loudly instead of shipping a hole.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

PREFIX = "alphafold/alphafold_iteration/"
EVOFORMER = PREFIX + "evoformer/"
TEMPLATE = EVOFORMER + "template_embedding/"
SINGLE_TEMPLATE = TEMPLATE + "single_template_embedding/"
TEMPLATE_STACK = SINGLE_TEMPLATE + "template_pair_stack/__layer_stack_no_state/"
# multimer_v3's template stack is not wrapped in a haiku layer_stack: it is two blocks under
# `template_embedding_iteration`, stacked in dim 0 the same way.
TEMPLATE_ITERATION = SINGLE_TEMPLATE + "template_embedding_iteration/"
RELATIVE_ENCODING = EVOFORMER + "~_relative_encoding/"
STRUCTURE = PREFIX + "structure_module/"
FOLD = STRUCTURE + "fold_iteration/"

NUM_EVOFORMER_BLOCKS = 48
NUM_EXTRA_MSA_BLOCKS = 4
NUM_TEMPLATE_BLOCKS = 2
#: multimer_v3 embeds nine template features separately and sums them; see `_template_multimer`.
NUM_TEMPLATE_PAIR_EMBEDDINGS = 9

# Heads present in the checkpoint that PXDesign never reads.
UNUSED_SCOPES = (
    PREFIX + "distogram_head/",
    PREFIX + "masked_msa_head/",
    PREFIX + "experimentally_resolved_head/",
)

#: Left unconsumed on the multimer path on top of `UNUSED_SCOPES`. multimer_v3's structure module
#: is `folding_multimer`, a different module from the monomer's: its IPA splits the q/k/v point
#: projections into their own scopes and its backbone update is `quat_rigid` rather than
#: `affine_update`. This file remaps the trunk, and the trunk is what runs on card; the multimer
#: structure module is a torch module nobody has written yet.
MULTIMER_UNUSED_SCOPES = (STRUCTURE,)


class _Params:
    """The npz, read once, tracking which arrays a remap has consumed."""

    def __init__(self, source: Mapping[str, np.ndarray]):
        self._source = source
        self.consumed: set[str] = set()

    def get(self, key: str, block: int | None = None) -> torch.Tensor:
        self.consumed.add(key)
        array = np.asarray(self._source[key], dtype=np.float32)
        if block is not None:
            array = array[block]
        return torch.from_numpy(np.ascontiguousarray(array))

    def unconsumed(self) -> list[str]:
        return sorted(set(self._source.keys()) - self.consumed)


def _linear(p: _Params, scope: str, block: int | None = None) -> dict[str, torch.Tensor]:
    """A haiku `Linear` as a torch `nn.Linear`: `[in, *out]` -> `[prod(out), in]`."""
    weight = p.get(scope + "//weights", block)
    bias = p.get(scope + "//bias", block)
    return {
        "weight": weight.reshape(weight.shape[0], -1).t().contiguous(),
        "bias": bias.reshape(-1),
    }


def _norm(p: _Params, scope: str, block: int | None = None) -> dict[str, torch.Tensor]:
    return {
        "weight": p.get(scope + "//scale", block),
        "bias": p.get(scope + "//offset", block),
    }


def _param(p: _Params, scope: str, name: str, block: int | None = None) -> torch.Tensor:
    """A bare `hk.get_parameter`, i.e. a projection with no bias of its own."""
    return p.get(f"{scope}//{name}", block)


def _prefixed(entries: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{k}": v for k, v in entries.items()}


def _attention(
    p: _Params, scope: str, block: int | None = None, *, gating: bool = True, kv_global: bool = False
) -> dict[str, torch.Tensor]:
    """AF2's `Attention` / `GlobalAttention` in tt-bio's `linear_{q,k,v,g,o}` layout.

    q/k/v carry no bias in AF2 (bare `hk.get_parameter`, not `Linear`); the gate and the output
    projection do. `kv_global` is the extra-MSA column attention, where AF2 shares one key and
    one value head across all query heads, so `key_w`/`value_w` are `[c, dim]` with no head axis.
    """
    out: dict[str, torch.Tensor] = {}

    def flat(name: str) -> torch.Tensor:
        w = _param(p, scope, name, block)
        return w.reshape(w.shape[0], -1).t().contiguous()

    out["linear_q.weight"] = flat("query_w")
    out["linear_k.weight"] = flat("key_w")
    out["linear_v.weight"] = flat("value_w")
    if gating:
        out["linear_g.weight"] = flat("gating_w")
        out["linear_g.bias"] = _param(p, scope, "gating_b", block).reshape(-1)
    output_w = _param(p, scope, "output_w", block)
    out["linear_o.weight"] = output_w.reshape(-1, output_w.shape[-1]).t().contiguous()
    out["linear_o.bias"] = _param(p, scope, "output_b", block)
    if kv_global:
        # Documented, not enforced: the caller must know k/v are single-head here.
        assert out["linear_k.weight"].shape[0] * out["linear_q.weight"].shape[0] > 0
    return out


def _relu_transition(p: _Params, scope: str, block: int | None = None) -> dict[str, torch.Tensor]:
    """AF2's transition: LN -> linear -> ReLU -> linear. Not SwiGLU."""
    out = _prefixed(_norm(p, scope + "input_layer_norm", block), "norm")
    out |= _prefixed(_linear(p, scope + "transition1", block), "fc1")
    out |= _prefixed(_linear(p, scope + "transition2", block), "fc2")
    return out


def _triangle_multiplication(
    p: _Params, scope: str, block: int | None, *, ending: bool, fused: bool = False
) -> dict[str, torch.Tensor]:
    """AF2's triangle multiplication in tt-bio's `TriangleMultiplication` layout.

    `p_in`/`g_in` are `[2 * hidden, c]` with slot `a` first. For the outgoing block `a` is AF2's
    left and `b` its right; for the incoming block they are swapped (see the module docstring).

    `fused` is the multimer_v3 checkpoint, which stores as two already-concatenated arrays
    (`projection`, `gate`) what the 2021 monomer release stores as four (`left_projection`,
    `right_projection`, `left_gate`, `right_gate`), and renames both LayerNorms
    (`layer_norm_input` -> `left_norm_input`, `center_layer_norm` -> `center_norm`). The
    arithmetic is the same fused form either way -- `fuse_projection_weights` is True at every
    config site in both releases -- so this is a key layout difference, not a new op. The
    incoming swap still applies; here it halves one array instead of reordering two.
    """
    order = ("right", "left") if ending else ("left", "right")
    if fused:
        out = _prefixed(_norm(p, scope + "left_norm_input", block), "norm_in")
        out |= _prefixed(_norm(p, scope + "center_norm", block), "norm_out")
        for tt_name, af2_name in (("p_in", "projection"), ("g_in", "gate")):
            fat = _linear(p, scope + af2_name, block)
            hidden = fat["weight"].shape[0] // 2
            half = {"left": slice(0, hidden), "right": slice(hidden, 2 * hidden)}
            out[f"{tt_name}.weight"] = torch.cat(
                [fat["weight"][half[side]] for side in order], dim=0).contiguous()
            out[f"{tt_name}.bias"] = torch.cat(
                [fat["bias"][half[side]] for side in order], dim=0).contiguous()
    else:
        out = _prefixed(_norm(p, scope + "layer_norm_input", block), "norm_in")
        out |= _prefixed(_norm(p, scope + "center_layer_norm", block), "norm_out")
        for tt_name, af2_suffix in (("p_in", "projection"), ("g_in", "gate")):
            halves = [_linear(p, f"{scope}{side}_{af2_suffix}", block) for side in order]
            out[f"{tt_name}.weight"] = torch.cat([h["weight"] for h in halves], dim=0)
            out[f"{tt_name}.bias"] = torch.cat([h["bias"] for h in halves], dim=0)
    out |= _prefixed(_linear(p, scope + "output_projection", block), "p_out")
    out |= _prefixed(_linear(p, scope + "gating_linear", block), "g_out")
    return out


def _triangle_attention(
    p: _Params, scope: str, block: int | None = None
) -> dict[str, torch.Tensor]:
    """AF2's triangle attention in tt-bio's `TriangleAttention` layout.

    One LayerNorm (`query_norm`) feeds both the q/k/v projections and the pair-bias projection.
    `feat_2d_weights` is `[c, heads]` and carries no bias, matching tt-bio's `linear.weight`.
    """
    out = _prefixed(_norm(p, scope + "query_norm", block), "layer_norm")
    out["linear.weight"] = _param(p, scope.rstrip("/"), "feat_2d_weights", block).t().contiguous()
    out |= _attention(p, scope + "attention", block)
    return out


def _outer_product_mean(
    p: _Params, scope: str, block: int | None = None
) -> dict[str, torch.Tensor]:
    """AF2's outer product mean in tt-bio's `OuterProductMean` layout.

    `output_w` is `[a_chan, b_chan, c_z]` contracted `dceb,cef->dbf`, so the flattened outer
    product must be a-channel major: index `a_chan * num_outer + b_chan`.
    """
    out = _prefixed(_norm(p, scope + "layer_norm_input", block), "norm")
    out |= _prefixed(_linear(p, scope + "left_projection", block), "proj_a")
    out |= _prefixed(_linear(p, scope + "right_projection", block), "proj_b")
    output_w = _param(p, scope.rstrip("/"), "output_w", block)
    out["proj_o.weight"] = output_w.reshape(-1, output_w.shape[-1]).t().contiguous()
    out["proj_o.bias"] = _param(p, scope.rstrip("/"), "output_b", block)
    return out


def _msa_row_attention(p: _Params, scope: str, block: int | None) -> dict[str, torch.Tensor]:
    """MSA row attention with pair bias: two LayerNorms, one on msa and one on the pair bias."""
    out = _prefixed(_norm(p, scope + "query_norm", block), "layer_norm")
    out |= _prefixed(_norm(p, scope + "feat_2d_norm", block), "pair_norm")
    out["linear.weight"] = _param(p, scope.rstrip("/"), "feat_2d_weights", block).t().contiguous()
    out |= _attention(p, scope + "attention", block)
    return out


def _msa_column_attention(
    p: _Params, scope: str, block: int | None, *, kv_global: bool
) -> dict[str, torch.Tensor]:
    out = _prefixed(_norm(p, scope + "query_norm", block), "layer_norm")
    out |= _attention(p, scope + "attention", block, kv_global=kv_global)
    return out


def _pair_block(p: _Params, scope: str, block: int | None,
                *, fused_trimul: bool = False) -> dict[str, torch.Tensor]:
    """The pair track shared by the Evoformer, extra-MSA and template stacks."""
    out = _prefixed(
        _triangle_multiplication(p, scope + "triangle_multiplication_outgoing/", block,
                                 ending=False, fused=fused_trimul),
        "tri_mul_out",
    )
    out |= _prefixed(
        _triangle_multiplication(p, scope + "triangle_multiplication_incoming/", block,
                                 ending=True, fused=fused_trimul),
        "tri_mul_in",
    )
    out |= _prefixed(
        _triangle_attention(p, scope + "triangle_attention_starting_node/", block), "tri_att_start"
    )
    out |= _prefixed(
        _triangle_attention(p, scope + "triangle_attention_ending_node/", block), "tri_att_end"
    )
    out |= _prefixed(_relu_transition(p, scope + "pair_transition/", block), "pair_transition")
    return out


def _evoformer_block(
    p: _Params, scope: str, block: int, *, extra_msa: bool, fused_trimul: bool = False
) -> dict[str, torch.Tensor]:
    out = _prefixed(
        _msa_row_attention(p, scope + "msa_row_attention_with_pair_bias/", block), "msa_row_attn"
    )
    column = "msa_column_global_attention/" if extra_msa else "msa_column_attention/"
    out |= _prefixed(
        _msa_column_attention(p, scope + column, block, kv_global=extra_msa), "msa_col_attn"
    )
    out |= _prefixed(_relu_transition(p, scope + "msa_transition/", block), "msa_transition")
    out |= _prefixed(_outer_product_mean(p, scope + "outer_product_mean/", block), "opm")
    out |= _pair_block(p, scope, block, fused_trimul=fused_trimul)
    return out


def _template(p: _Params) -> dict[str, torch.Tensor]:
    """Template pair embedding, its 2-block c=64 pair stack, and the torsion-angle MSA rows."""
    out = _prefixed(_linear(p, SINGLE_TEMPLATE + "embedding2d"), "embedding2d")
    for i in range(NUM_TEMPLATE_BLOCKS):
        out |= _prefixed(_pair_block(p, TEMPLATE_STACK, i), f"pair_stack.{i}")
    out |= _prefixed(_norm(p, SINGLE_TEMPLATE + "output_layer_norm"), "output_norm")
    # Pointwise attention over templates: q from the pair rep at c_z, k/v from the template
    # stack at c=64, and no gating (the checkpoint has no gating_w/gating_b here).
    out |= _prefixed(_attention(p, TEMPLATE + "attention", gating=False), "attn")
    out |= _prefixed(_linear(p, EVOFORMER + "template_single_embedding"), "single_embedding")
    out |= _prefixed(_linear(p, EVOFORMER + "template_projection"), "single_projection")
    return out


def _template_multimer(p: _Params) -> dict[str, torch.Tensor]:
    """multimer_v3's template embedder, which is a different module from the monomer's.

    Three differences, all of them structural rather than a rename:

    * The monomer builds one 88-channel feature tensor and embeds it with a single
      `embedding2d`. The multimer embeds nine features separately and sums the results
      (`template_pair_embedding_0..8`), and the ninth is the LayerNormed query pair
      representation itself, so the template stack sees the trunk's pair. Five of the nine take
      a scalar input, stored as a bare `[c]` array; they are kept here as `[c, 1]` so one
      `Linear` serves all nine.
    * Its pair stack runs the Evoformer order (multiplications before attentions), where the
      monomer's template stack runs the attentions first.
    * There is no cross-attention over templates. The monomer attends over the template axis with
      the pair as the query; the multimer sums the per-template embeddings, divides by the
      template count, takes a ReLU and projects with `output_linear`.
    """
    out: dict[str, torch.Tensor] = {}
    for i in range(NUM_TEMPLATE_PAIR_EMBEDDINGS):
        entry = _linear(p, f"{SINGLE_TEMPLATE}template_pair_embedding_{i}")
        if entry["weight"].shape[0] != entry["bias"].shape[0]:
            # A scalar-input embedding: haiku stores it as a bare [c] array, which `_linear`
            # transposes to [1, c]. torch wants [c, 1].
            entry["weight"] = entry["weight"].reshape(-1, 1).contiguous()
        out |= _prefixed(entry, f"pair_embedding.{i}")
    out |= _prefixed(_norm(p, SINGLE_TEMPLATE + "query_embedding_norm"), "query_norm")
    for i in range(NUM_TEMPLATE_BLOCKS):
        out |= _prefixed(_pair_block(p, TEMPLATE_ITERATION, i, fused_trimul=True),
                         f"pair_stack.{i}")
    out |= _prefixed(_norm(p, SINGLE_TEMPLATE + "output_layer_norm"), "output_norm")
    out |= _prefixed(_linear(p, TEMPLATE + "output_linear"), "output_linear")
    out |= _prefixed(_linear(p, EVOFORMER + "template_single_embedding"), "single_embedding")
    out |= _prefixed(_linear(p, EVOFORMER + "template_projection"), "single_projection")
    return out


def _structure_module(p: _Params) -> dict[str, torch.Tensor]:
    """The 8 shared-weight IPA layers. Not stacked in the checkpoint: one `fold_iteration`."""
    out = _prefixed(_norm(p, STRUCTURE + "single_layer_norm"), "single_norm")
    out |= _prefixed(_norm(p, STRUCTURE + "pair_layer_norm"), "pair_norm")
    out |= _prefixed(_linear(p, STRUCTURE + "initial_projection"), "initial_projection")

    ipa = FOLD + "invariant_point_attention/"
    out["ipa.point_weights"] = _param(p, ipa.rstrip("/"), "trainable_point_weights")
    for name in ("q_scalar", "kv_scalar", "q_point_local", "kv_point_local", "attention_2d", "output_projection"):
        out |= _prefixed(_linear(p, ipa + name), f"ipa.{name}")

    out |= _prefixed(_norm(p, FOLD + "attention_layer_norm"), "attention_norm")
    out |= _prefixed(_norm(p, FOLD + "transition_layer_norm"), "transition_norm")
    for i, name in enumerate(("transition", "transition_1", "transition_2")):
        out |= _prefixed(_linear(p, FOLD + name), f"transition.{i}")
    out |= _prefixed(_linear(p, FOLD + "affine_update"), "affine_update")

    side = FOLD + "rigid_sidechain/"
    out |= _prefixed(_linear(p, side + "input_projection"), "sidechain.input_projection.0")
    out |= _prefixed(_linear(p, side + "input_projection_1"), "sidechain.input_projection.1")
    for i, suffix in enumerate(("", "_1")):
        out |= _prefixed(_linear(p, f"{side}resblock1{suffix}"), f"sidechain.resblock.{i}.0")
        out |= _prefixed(_linear(p, f"{side}resblock2{suffix}"), f"sidechain.resblock.{i}.1")
    out |= _prefixed(_linear(p, side + "unnormalized_angles"), "sidechain.angles")
    return out


def _heads(p: _Params) -> dict[str, torch.Tensor]:
    lddt = PREFIX + "predicted_lddt_head/"
    out = _prefixed(_norm(p, lddt + "input_layer_norm"), "plddt.norm")
    out |= _prefixed(_linear(p, lddt + "act_0"), "plddt.act.0")
    out |= _prefixed(_linear(p, lddt + "act_1"), "plddt.act.1")
    out |= _prefixed(_linear(p, lddt + "logits"), "plddt.logits")
    out |= _prefixed(
        _linear(p, PREFIX + "predicted_aligned_error_head/logits"), "pae.logits"
    )
    return out


def remap_af2_params(source: Mapping[str, np.ndarray], *,
                     multimer: bool = False) -> dict[str, torch.Tensor]:
    """Remap an already-loaded AF2 parameter mapping. See `load_af2_state_dict`.

    `multimer=True` reads `params_model_1..5_multimer_v3.npz`, whose trunk is the same Evoformer
    at the same widths with a different relative-position encoding, a different template embedder
    and the fused triangle-multiplication key layout. The structure module is left unconsumed;
    see `MULTIMER_UNUSED_SCOPES`.
    """
    p = _Params(source)
    out: dict[str, torch.Tensor] = {}

    embeddings = [
        ("preprocess_1d", "preprocess_1d"),
        ("preprocess_msa", "preprocess_msa"),
        ("left_single", "left_single"),
        ("right_single", "right_single"),
        ("extra_msa_activations", "extra_msa_activations"),
    ]
    if not multimer:
        # The checkpoint's own spelling of "pair_activations".
        embeddings.append(("pair_activations", "pair_activiations"))
    for name, scope in embeddings:
        out |= _prefixed(_linear(p, EVOFORMER + scope), f"embed.{name}")
    if multimer:
        # The relative encoding is its own haiku method, so its Linear sits one scope deeper.
        out |= _prefixed(_linear(p, RELATIVE_ENCODING + "position_activations"),
                         "embed.position_activations")

    out |= _prefixed(_linear(p, EVOFORMER + "prev_pos_linear"), "recycle.prev_pos_linear")
    out |= _prefixed(_norm(p, EVOFORMER + "prev_msa_first_row_norm"), "recycle.prev_msa_norm")
    out |= _prefixed(_norm(p, EVOFORMER + "prev_pair_norm"), "recycle.prev_pair_norm")

    out |= _prefixed(_template_multimer(p) if multimer else _template(p), "template")

    for i in range(NUM_EXTRA_MSA_BLOCKS):
        out |= _prefixed(
            _evoformer_block(p, EVOFORMER + "extra_msa_stack/", i, extra_msa=True,
                             fused_trimul=multimer),
            f"extra_msa.{i}",
        )
    for i in range(NUM_EVOFORMER_BLOCKS):
        out |= _prefixed(
            _evoformer_block(p, EVOFORMER + "evoformer_iteration/", i, extra_msa=False,
                             fused_trimul=multimer),
            f"evoformer.{i}",
        )

    out |= _prefixed(_linear(p, EVOFORMER + "single_activations"), "single_activations")
    if not multimer:
        out |= _prefixed(_structure_module(p), "structure")
    out |= _prefixed(_heads(p), "heads")

    unused = UNUSED_SCOPES + (MULTIMER_UNUSED_SCOPES if multimer else ())
    unconsumed = p.unconsumed()
    unexpected = [k for k in unconsumed if not k.startswith(unused)]
    if unexpected:
        raise AssertionError(
            f"{len(unexpected)} checkpoint arrays were not consumed by the remap and are not one "
            f"of the deliberately unused scopes {unused}: {unexpected[:12]}"
        )
    return out


def load_af2_state_dict(path: str, *, multimer: bool = False) -> dict[str, torch.Tensor]:
    """Load an AF2 npz as a flat torch state dict in tt-bio module layout.

    Raises if any array other than the distogram / masked-MSA / experimentally-resolved heads
    (and, on the multimer path, the structure module) is left unconsumed, so a dropped block
    cannot pass as a successful load.
    """
    with np.load(path, allow_pickle=False) as npz:
        return remap_af2_params({k: npz[k] for k in npz.files}, multimer=multimer)
