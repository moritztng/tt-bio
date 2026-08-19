"""Map RF3 checkpoint weight names onto tt-bio's shared ttnn blocks.

RF3's Pairformer is AF3-shaped and so are tt-bio's `TriangleMultiplication`,
`TriangleAttention`, `AttentionPairBias` and `Transition` in `tt_bio.tenstorrent`.
The maths lines up; only the names differ, and for triangle multiplication not even
those. So the port reuses the shared blocks and remaps, rather than reimplementing.

Two places the mapping is more than a rename:

- RF3's attention normalises its single input inside the module (`ln_1`), while
  tt-bio's `PairformerLayer` normalises outside as `pre_norm_s`. Same computation,
  different owner, so `ln_1` moves out of the attention scope.
- RF3's q/k/v projections carry no bias; tt-bio's non-atom-level `AttentionPairBias`
  reads `proj_q.bias` unconditionally. The remap synthesises a zero bias, which is
  what "no bias" means numerically.

`check_coverage` is card-free: it verifies every RF3 key is consumed and every key
the shared blocks ask for is produced, which is the half of a weight remap that
silently goes wrong.
"""

from __future__ import annotations

import torch

#: RF3 block-relative name -> tt-bio block-relative name.
#: Triangle multiplication is absent because its six weights already match.
PAIRFORMER_RENAMES: dict[str, str] = {
    # scopes
    "tri_mul_outgoing": "tri_mul_out",
    "tri_mul_incoming": "tri_mul_in",
    "tri_attn_start": "tri_att_start",
    "tri_attn_end": "tri_att_end",
    "z_transition": "transition_z",
    "s_transition": "transition_s",
    "attention_pair_bias": "attention",
}

#: leaf renames, applied within a scope
TRI_ATT_LEAVES = {
    "norm": "layer_norm",
    "to_q": "linear_q",
    "to_k": "linear_k",
    "to_v": "linear_v",
    "to_g": "linear_g",
    "to_out": "linear_o",
    "to_b": "linear",
}
TRANSITION_LEAVES = {
    "layer_norm_1": "norm",
    "linear_1": "fc1",
    "linear_2": "fc2",
    "linear_3": "fc3",
}
ATTENTION_LEAVES = {
    "to_q": "proj_q",
    "to_k": "proj_k",
    "to_v": "proj_v",
    "to_g.0": "proj_g",
    "to_a": "proj_o",
    "ln_0": "proj_z.0",
    "to_b": "proj_z.1",
}


def _leaf(name: str) -> tuple[str, str]:
    """Split a block-relative key into (scope, remainder)."""
    head, _, rest = name.partition(".")
    return head, rest


def remap_pairformer_block(block: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap one RF3 Pairformer block's weights to tt-bio's names.

    Args:
        block: block-relative RF3 weights, e.g. ``tri_mul_outgoing.p_in.weight``.

    Returns:
        block-relative tt-bio weights, e.g. ``tri_mul_out.p_in.weight``.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in block.items():
        scope, rest = _leaf(key)
        new_scope = PAIRFORMER_RENAMES.get(scope, scope)

        if scope in ("tri_mul_outgoing", "tri_mul_incoming"):
            out[f"{new_scope}.{rest}"] = value  # names already match
        elif scope in ("tri_attn_start", "tri_attn_end"):
            out[f"{new_scope}.{_sub(rest, TRI_ATT_LEAVES)}"] = value
        elif scope in ("z_transition", "s_transition"):
            out[f"{new_scope}.{_sub(rest, TRANSITION_LEAVES)}"] = value
        elif scope == "attention_pair_bias":
            if rest.startswith("ln_1."):
                # tt-bio's PairformerLayer owns this norm, not the attention block
                out["pre_norm_s." + rest[len("ln_1."):]] = value
            else:
                out[f"{new_scope}.{_sub(rest, ATTENTION_LEAVES)}"] = value
        else:
            raise KeyError(f"unmapped RF3 pairformer scope: {scope!r} (from {key!r})")

    # RF3's q/k/v are bias-free; tt-bio reads proj_q.bias unconditionally.
    qw = out.get("attention.proj_q.weight")
    if qw is not None and "attention.proj_q.bias" not in out:
        out["attention.proj_q.bias"] = torch.zeros(
            qw.shape[0], dtype=qw.dtype, device=qw.device
        )
    return out


def _sub(rest: str, leaves: dict[str, str]) -> str:
    """Rename the longest matching leaf prefix of ``rest``."""
    for src in sorted(leaves, key=len, reverse=True):
        if rest == src or rest.startswith(src + "."):
            return leaves[src] + rest[len(src):]
    return rest


#: What the shared blocks read, per tt-bio `PairformerLayer(transform_s=True)`.
EXPECTED = {
    "pre_norm_s.weight", "pre_norm_s.bias",
    "attention.proj_q.weight", "attention.proj_q.bias",
    "attention.proj_k.weight", "attention.proj_v.weight",
    "attention.proj_g.weight", "attention.proj_o.weight",
    "attention.proj_z.0.weight", "attention.proj_z.0.bias",
    "attention.proj_z.1.weight",
    "transition_s.norm.weight", "transition_s.norm.bias",
    "transition_s.fc1.weight", "transition_s.fc2.weight", "transition_s.fc3.weight",
    "transition_z.norm.weight", "transition_z.norm.bias",
    "transition_z.fc1.weight", "transition_z.fc2.weight", "transition_z.fc3.weight",
}
for _s in ("tri_mul_out", "tri_mul_in"):
    EXPECTED |= {f"{_s}.{n}" for n in (
        "norm_in.weight", "norm_in.bias", "norm_out.weight", "norm_out.bias",
        "g_in.weight", "g_out.weight", "p_in.weight", "p_out.weight",
    )}
for _s in ("tri_att_start", "tri_att_end"):
    EXPECTED |= {f"{_s}.{n}" for n in (
        "layer_norm.weight", "layer_norm.bias", "linear_q.weight", "linear_k.weight",
        "linear_v.weight", "linear_g.weight", "linear_o.weight", "linear.weight",
    )}


def check_coverage(block: dict[str, torch.Tensor]) -> dict:
    """Card-free check that the remap is complete in both directions."""
    got = remap_pairformer_block(block)
    produced = set(got)
    missing = sorted(EXPECTED - produced)
    extra = sorted(produced - EXPECTED)
    return {
        "rf3_keys": len(block),
        "produced": len(produced),
        "expected": len(EXPECTED),
        "missing": missing,
        "extra": extra,
        "ok": not missing,
    }


# --- MSA module -------------------------------------------------------------
# The MSA module spells triangle multiplication `tri_mult_*` where the Pairformer
# stack spells it `tri_mul_*`, so PAIRFORMER_RENAMES silently would not match. It
# also has no block index: one set of weights drives all n_block iterations.
MSA_SCOPES: dict[str, str] = {
    "msa_subsampler": "msa_subsampler",
    "outer_product": "outer_product",
    "msa_pair_weighted_averaging": "msa_pair_weighted_averaging",
    "msa_transition": "msa_transition",
    # the inner pairformer's five sub-blocks all live under one scope in the port
    "tri_mult_outgoing": "pairformer_layer.tri_mul_out",
    "tri_mult_incoming": "pairformer_layer.tri_mul_in",
    "tri_attn_start": "pairformer_layer.tri_att_start",
    "tri_attn_end": "pairformer_layer.tri_att_end",
    "pair_transition": "pairformer_layer.transition_z",
}

OUTER_PRODUCT_LEAVES = {
    "proj_left": "proj_a",
    "proj_right": "proj_b",
    "proj_out": "proj_o",
}
PWA_LEAVES = {
    "norm_msa": "norm_m",
    "norm_pair": "norm_z",
    "to_v": "proj_m",
    "to_bias": "proj_z",
    "to_gate": "proj_g",
    "to_out": "proj_o",
}


def remap_msa_module(block: dict) -> dict:
    """Remap RF3's `recycler.msa_module` weights onto the ported MSA module.

    Args:
        block: module-relative RF3 weights, e.g. ``outer_product.proj_left.weight``.
    """
    out: dict = {}
    for key, value in block.items():
        scope, rest = _leaf(key)
        if scope not in MSA_SCOPES:
            raise KeyError(f"unmapped RF3 msa_module scope: {scope!r} (from {key!r})")
        new_scope = MSA_SCOPES[scope]

        if scope == "outer_product":
            rest = _sub(rest, OUTER_PRODUCT_LEAVES)
        elif scope == "msa_pair_weighted_averaging":
            rest = _sub(rest, PWA_LEAVES)
        elif scope in ("msa_transition", "pair_transition"):
            rest = _sub(rest, TRANSITION_LEAVES)
        elif scope in ("tri_attn_start", "tri_attn_end"):
            rest = _sub(rest, TRI_ATT_LEAVES)
        # tri_mult_* leaves and msa_subsampler leaves already match

        out[f"{new_scope}.{rest}"] = value
    return out
