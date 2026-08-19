#!/usr/bin/env python3
"""Report RF3 weights that tt-bio's shared blocks do not read.

The recurring failure mode on this port is a semantic feature RF3 has that the
shared block does not: triangle-attention gate/output biases, outer-product
left/right biases, the ending pair-bias transpose, fp32 softmax. Each was found
after the fact, twice by a PCC that looked fine.

This is the pre-flight instead. For a checkpoint prefix and the tt-bio block that
would be reused, it lists the keys RF3 carries that the block never reads, with
magnitudes, so an unsupported feature is a decision before any code is written
rather than a bisect afterwards.

    python scripts/rf3_port/check_block_support.py --ckpt /path/to/rf3_latest.ckpt
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch

#: What each tt-bio shared block actually reads, taken from tt_bio/tenstorrent.py.
TT_BIO_READS = {
    "TriangleMultiplication": {
        "norm_in.weight", "norm_in.bias", "norm_out.weight", "norm_out.bias",
        "g_in.weight", "g_out.weight", "p_in.weight", "p_out.weight",
    },
    "TriangleAttention": {
        "layer_norm.weight", "layer_norm.bias", "linear_q.weight", "linear_k.weight",
        "linear_v.weight", "linear_g.weight", "linear_g.bias", "linear_o.weight",
        "linear_o.bias", "linear.weight",
    },
    "AttentionPairBias": {
        "proj_q.weight", "proj_q.bias", "proj_k.weight", "proj_v.weight",
        "proj_g.weight", "proj_o.weight", "proj_z.0.weight", "proj_z.0.bias",
        "proj_z.1.weight",
    },
    # The AF3 DiT block. It owns an AdaLN, an AttentionPairBias, the adaLN-Zero
    # output gate and a ConditionedTransitionBlock, so an RF3 atom-transformer
    # block must be compared against THIS, not against AttentionPairBias alone.
    "DiffusionTransformerLayer": {
        "output_projection_linear.weight", "output_projection_linear.bias",
    } | {f"adaln.{k}" for k in
         ("s_norm.weight", "s_bias.weight", "s_scale.weight", "s_scale.bias")}
      | {f"pair_bias_attn.{k}" for k in
         ("proj_q.weight", "proj_q.bias", "proj_k.weight", "proj_v.weight",
          "proj_g.weight", "proj_o.weight", "proj_z.0.weight", "proj_z.0.bias",
          "proj_z.1.weight")},
    "AdaLN": {"s_norm.weight", "s_bias.weight", "s_scale.weight", "s_scale.bias"},
    "ConditionedTransitionBlock": {
        "a_to_b.weight", "b_to_a.weight", "swish_gate.0.weight",
        "output_projection.0.weight", "output_projection.0.bias",
    },
    "Transition": {"norm.weight", "norm.bias", "fc1.weight", "fc2.weight", "fc3.weight"},
    "OuterProductMean": {
        "norm.weight", "norm.bias", "proj_a.weight", "proj_a.bias",
        "proj_b.weight", "proj_b.bias", "proj_o.weight", "proj_o.bias",
    },
    "PairWeightedAveraging": {
        "norm_m.weight", "norm_m.bias", "norm_z.weight", "norm_z.bias",
        "proj_m.weight", "proj_z.weight", "proj_g.weight", "proj_o.weight",
    },
}

#: RF3 sub-module leaf -> the tt-bio block that would be reused for it, plus the
#: leaf renames needed to compare like with like.
TARGETS = [
    # RF3's atom-transformer block is the AF3 DiT block: AdaLN in, attention,
    # adaLN-Zero output gate from S, conditioned transition. Compare it against
    # tt-bio's DiffusionTransformerLayer as a whole.
    ("feature_initializer.input_feature_embedder.atom_attention_encoder"
     ".atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias",
     "DiffusionTransformerLayer",
     {"to_q": "pair_bias_attn.proj_q", "to_k": "pair_bias_attn.proj_k",
      "to_v": "pair_bias_attn.proj_v", "to_g.0": "pair_bias_attn.proj_g",
      "to_a": "pair_bias_attn.proj_o", "ln_0": "pair_bias_attn.proj_z.0",
      "to_b": "pair_bias_attn.proj_z.1",
      "linear_output_project.0": "output_projection_linear",
      "ada_ln_1.ln_s": "adaln.s_norm", "ada_ln_1.to_bias": "adaln.s_bias",
      "ada_ln_1.to_gain.0": "adaln.s_scale"}),
    ("feature_initializer.input_feature_embedder.atom_attention_encoder"
     ".atom_transformer.diffusion_transformer.blocks.0.conditioned_transition_block",
     "ConditionedTransitionBlock",
     {"linear_1": "a_to_b", "linear_2": "swish_gate.0", "linear_3": "b_to_a",
      "linear_output_project.0": "output_projection.0"}),
]


#: Sub-module names scored under their own entry, so their keys must not be
#: re-reported as gaps under the parent. ConditionedTransitionBlock nests an
#: AdaLN that maps cleanly on its own.
NESTED_BLOCKS = {"ada_ln"}


def rename(leaf: str, table: dict) -> str:
    for src in sorted(table, key=len, reverse=True):
        if leaf == src or leaf.startswith(src + "."):
            return table[src] + leaf[len(src):]
    return leaf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    rows = []
    for prefix, block, table in TARGETS:
        full = f"shadow.{prefix}."
        have = {k[len(full):]: v for k, v in sd.items() if k.startswith(full)}
        # Only this level. A nested sub-module is scored under its own entry, or --
        # like ConditionedTransitionBlock's AdaLN -- is the same block type already
        # covered elsewhere. Reporting its keys here would double-count them as gaps.
        have = {k: v for k, v in have.items()
                if k.split(".")[0] not in NESTED_BLOCKS}
        reads = TT_BIO_READS[block]

        unsupported = []
        for leaf, tensor in sorted(have.items()):
            if rename(leaf, table) not in reads:
                unsupported.append({
                    "key": leaf,
                    "shape": list(tensor.shape),
                    "absmax": round(float(tensor.abs().max()), 4),
                    "all_zero": bool(torch.count_nonzero(tensor) == 0),
                })
        rows.append({
            "rf3_module": prefix.split(".")[-1] if "." in prefix else prefix,
            "tt_bio_block": block,
            "rf3_keys": len(have),
            "unsupported": unsupported,
        })

    print(json.dumps({"targets": rows}, indent=2))
    n = sum(len(r["unsupported"]) for r in rows)
    print(f"{'GAP' if n else 'OK'}: {n} RF3 weights unread by the shared blocks")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
