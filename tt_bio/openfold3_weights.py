"""OpenFold3 -> tt-bio weight-name remaps (pure dict/tensor functions).

OpenFold3 is the same AlphaFold3 family as Protenix-v2 and Boltz-2, so its trunk maps
onto the same tt_bio.tenstorrent primitives. The math is identical; only the checkpoint
key names differ. So instead of duplicating the remap logic, each function here renames
OF3 keys onto the Protenix-v2 key names and delegates to the proven, on-device-validated
remaps in protenix_weights.py (PCC > 0.98; see tests/test_openfold3_*.py).

OF3 PairFormerBlock vs Protenix-v2 block key deltas (structurally identical modules):
  - pair ops nested under `pair_stack.`      (Protenix: top level)
  - `attn_pair_bias`                          (Protenix: `attention_pair_bias`)
  - SwiGLU transition `layer_norm`/`swiglu.linear_a`/`swiglu.linear_b`/`linear_out`
      (Protenix: `layernorm1`/`linear_no_bias_a`/`linear_no_bias_b`/`linear_no_bias`)
  - TriangleAttention bias proj `linear_z`    (Protenix: `linear`)
  - AttentionPairBias `mha.*`/`layer_norm_a`/`layer_norm_z`/`linear_z`
      (Protenix: `attention.*`/`layernorm_a`/`layernorm_z`/`linear_nobias_z`)
TriangleMultiplication keys are byte-identical to Protenix (no rename needed).

No openfold3 import -- pure torch rename on tensors.
"""

from __future__ import annotations

from . import protenix_weights as pw


def _sub(sd: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _rename_transition(sd: dict) -> dict:
    """OF3 SwiGLUTransition -> Protenix transition key names."""
    return {
        "layernorm1.weight": sd["layer_norm.weight"],
        "layernorm1.bias": sd["layer_norm.bias"],
        "linear_no_bias_a.weight": sd["swiglu.linear_a.weight"],
        "linear_no_bias_b.weight": sd["swiglu.linear_b.weight"],
        "linear_no_bias.weight": sd["linear_out.weight"],
    }


def _rename_tri_att(sd: dict) -> dict:
    """OF3 TriangleAttention -> Protenix (only the bias proj differs)."""
    out = {k: v for k, v in sd.items() if k != "linear_z.weight"}
    out["linear.weight"] = sd["linear_z.weight"]
    return out


def _rename_attention_pair_bias(sd: dict) -> dict:
    """OF3 AttentionPairBias (use_ada_layer_norm=False) -> Protenix key names."""
    out = {
        "layernorm_a.weight": sd["layer_norm_a.weight"],
        "layernorm_a.bias": sd["layer_norm_a.bias"],
        "layernorm_z.weight": sd["layer_norm_z.weight"],
        "layernorm_z.bias": sd["layer_norm_z.bias"],
        "linear_nobias_z.weight": sd["linear_z.weight"],
    }
    for k, v in sd.items():
        if k.startswith("mha."):
            out["attention." + k[len("mha."):]] = v
    return out


def remap_pairformer_block(block_sd: dict) -> dict:
    """OF3 PairFormerBlock state_dict -> tt-bio PairformerLayer flat state_dict.

    block_sd keys are stripped of the `pairformer_stack.blocks.{i}.` prefix.
    """
    pd: dict = {}
    for name in ("tri_mul_out", "tri_mul_in"):                       # identical keys
        for k, v in _sub(block_sd, f"pair_stack.{name}").items():
            pd[f"{name}.{k}"] = v
    for name in ("tri_att_start", "tri_att_end"):
        for k, v in _rename_tri_att(_sub(block_sd, f"pair_stack.{name}")).items():
            pd[f"{name}.{k}"] = v
    for k, v in _rename_transition(_sub(block_sd, "pair_stack.pair_transition")).items():
        pd[f"pair_transition.{k}"] = v
    for k, v in _rename_attention_pair_bias(_sub(block_sd, "attn_pair_bias")).items():
        pd[f"attention_pair_bias.{k}"] = v
    for k, v in _rename_transition(_sub(block_sd, "single_transition")).items():
        pd[f"single_transition.{k}"] = v
    return pw.remap_pairformer_block(pd)


def remap_pairformer_stack(sd: dict, prefix: str = "pairformer_stack") -> dict:
    """Full 48-block OF3 pairformer_stack -> tt-bio Pairformer `layers.{i}.*` dict."""
    import re
    pat = re.compile(rf"^{re.escape(prefix)}\.blocks\.(\d+)\.")
    nb = 1 + max(int(pat.match(k).group(1)) for k in sd if pat.match(k))
    combined: dict = {}
    for i in range(nb):
        block_sd = _sub(sd, f"{prefix}.blocks.{i}")
        for k, v in remap_pairformer_block(block_sd).items():
            combined[f"layers.{i}.{k}"] = v
    return combined


def _rename_pair_weighted_averaging(sd: dict) -> dict:
    """OF3 MSAPairWeightedAveraging -> Protenix MSAPairWeightedAveraging key names."""
    return {
        "layernorm_m.weight": sd["layer_norm_m.weight"],
        "layernorm_m.bias": sd["layer_norm_m.bias"],
        "layernorm_z.weight": sd["layer_norm_z.weight"],
        "layernorm_z.bias": sd["layer_norm_z.bias"],
        "linear_no_bias_mv.weight": sd["linear_v.weight"],
        "linear_no_bias_mg.weight": sd["linear_g.weight"],
        "linear_no_bias_z.weight": sd["linear_z.weight"],
        "linear_no_bias_out.weight": sd["linear_o.weight"],
    }


def remap_msa_block(block_sd: dict) -> dict:
    """OF3 MSAModuleBlock state_dict -> raw-primitive state dicts, one per tt-bio
    primitive (OuterProductMean, PairWeightedAveraging, Transition, PairformerLayer
    pair-only), NOT a single combined dict.

    `block_sd` keys are stripped of the `msa_module.blocks.{i}.` prefix.

    OF3 (like Protenix-v2) runs opm_first=True: OuterProductMean happens BEFORE the
    MSA update, the reverse of tt_bio.tenstorrent.MSALayer's hardcoded order (which
    matches Boltz-2's opm_first=False convention). So callers must compose these
    primitives directly in OF3's order (mirrors tests/test_protenix_trunk_msa.py) --
    MSALayer would silently apply the wrong order. See docs/openfold3-port.md.

    The last block (`last_block=True` in the reference, `opm_first=True` ->
    `skip_msa_update=True`) has no `msa_att_row`/`msa_transition` keys; the returned
    dict then omits "pair_weighted_averaging"/"msa_transition" accordingly.
    """
    out = {"outer_product_mean": pw.remap_outer_product_mean(_sub(block_sd, "outer_product_mean"))}

    pair_stack_sd: dict = {}
    for name in ("tri_mul_out", "tri_mul_in"):
        for k, v in _sub(block_sd, f"pair_stack.{name}").items():
            pair_stack_sd[f"{name}.{k}"] = v
    for name in ("tri_att_start", "tri_att_end"):
        for k, v in _rename_tri_att(_sub(block_sd, f"pair_stack.{name}")).items():
            pair_stack_sd[f"{name}.{k}"] = v
    for k, v in _rename_transition(_sub(block_sd, "pair_stack.pair_transition")).items():
        pair_stack_sd[f"pair_transition.{k}"] = v
    out["pair_stack"] = pw.remap_msa_pair_stack(pair_stack_sd)

    if any(k.startswith("msa_att_row.") for k in block_sd):
        out["pair_weighted_averaging"] = pw.remap_pair_weighted_averaging(
            _rename_pair_weighted_averaging(_sub(block_sd, "msa_att_row")))
        out["msa_transition"] = pw.remap_transition(_rename_transition(_sub(block_sd, "msa_transition")))
    return out


def remap_msa_module(sd: dict, prefix: str = "msa_module") -> dict:
    """Full OF3 msa_module -> list of per-block remapped primitive dicts (see
    `remap_msa_block`), in block order. Block count and the last block's
    skip_msa_update are inferred from which keys are present."""
    import re
    pat = re.compile(rf"^{re.escape(prefix)}\.blocks\.(\d+)\.")
    nb = 1 + max(int(pat.match(k).group(1)) for k in sd if pat.match(k))
    return [remap_msa_block(_sub(sd, f"{prefix}.blocks.{i}")) for i in range(nb)]
