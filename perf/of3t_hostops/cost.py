#!/usr/bin/env python3
"""D127 Deliverable 3 -- cost the two atom-encoder ports by splitting each scope into the part
that needs a blocked gather and the part that does not.

`ref_atom_embed` (openfold3_host_prep.py:180) builds two things from eight bias-free linears:

  cl  -- five linears on PER-ATOM features (ref_pos, ref_charge, ref_mask, ref_element,
         ref_atom_name_chars), summed. Every one is `x @ w.t()` on a [n_atom, d] tensor. No
         blocking, no gather, no index arithmetic.
  plm -- three linears (ref_offset, inv_sq_dists, valid_mask) on the BLOCKED pair
         representation, which is where `convert_single_rep_to_blocks` and `get_block_indices`
         live, and which is the part ttnn's per-element gather makes expensive.

Whether that distinction matters is an empirical question about where the gradient mass is, and
that is what this answers. Same for `run_input_atom_encoder`, split into its pair-completion
leg (blocked) and its `linear_q` aggregation leg (a matmul, a relu and a second matmul).
"""
import json, os, torch

REF = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
ref = torch.load(REF, map_location="cpu", weights_only=False)
sq = {k: float((v.double() ** 2).sum()) for k, v in ref.items() if torch.is_tensor(v)}
TOTAL = sum(sq.values())

CL = ("linear_ref_pos", "linear_ref_charge", "linear_ref_mask", "linear_ref_element",
      "linear_ref_atom_chars")
PLM = ("linear_ref_offset", "linear_inv_sq_dists", "linear_valid_mask")

groups = []
for enc in ("diffusion_module.atom_attn_enc", "input_embedder.atom_attn_enc"):
    rafe = enc + ".ref_atom_feature_embedder."
    groups.append((enc + " ref_atom_embed -> cl (per-atom linears, no blocking)",
                   [rafe + n + ".weight" for n in CL], "matmul only"))
    groups.append((enc + " ref_atom_embed -> plm (blocked pair rep)",
                   [rafe + n + ".weight" for n in PLM], "needs the blocked gather"))

ie = "input_embedder.atom_attn_enc."
groups.append((ie + "pair completion (linear_l, linear_m, pair_mlp)",
               [ie + n for n in ("linear_l.weight", "linear_m.weight", "pair_mlp.1.weight",
                                 "pair_mlp.3.weight", "pair_mlp.5.weight")],
               "needs the blocked gather"))
groups.append((ie + "linear_q aggregation leg",
               [ie + "linear_q.0.weight"], "matmul + relu + matmul"))
groups.append((ie + "atom_transformer (already on device inside run_input_atom_encoder)",
               [k for k in sq if k.startswith(ie + "atom_transformer.")], "already device"))

print("%-74s %6s %9s  %s" % ("group", "n", "share", "what it needs"))
tot_easy = tot_hard = 0.0
for label, names, need in groups:
    present = [n for n in names if n in sq]
    m = sum(sq[n] for n in present)
    print("%-74s %6d %8.4f%%  %s" % (label, len(present), 100 * m / TOTAL, need))
    if need == "matmul only" or need.startswith("matmul +"):
        tot_easy += m
    elif need == "needs the blocked gather":
        tot_hard += m
print("\nmass on ops that are matmul/relu only:      %8.4f %%" % (100 * tot_easy / TOTAL))
print("mass behind the blocked gather:            %8.4f %%" % (100 * tot_hard / TOTAL))
json.dump({"pct_matmul_only": 100 * tot_easy / TOTAL,
           "pct_behind_blocked_gather": 100 * tot_hard / TOTAL,
           "groups": [{"group": l, "n": len([n for n in ns if n in sq]),
                       "pct": 100 * sum(sq[n] for n in ns if n in sq) / TOTAL, "needs": w}
                      for l, ns, w in groups]},
          open("perf/of3t_hostops/COST.json", "w"), indent=1)
print("wrote perf/of3t_hostops/COST.json")
