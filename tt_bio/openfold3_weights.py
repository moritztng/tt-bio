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
