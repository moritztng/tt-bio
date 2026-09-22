#!/usr/bin/env python3
"""The matched scope, by mass, before anything is scored.

The device arm places 2,496 of the pairformer stack`s 2,736 tensors and leaves 240 unplaced.
A count says 91.2 % covered; a count is not the quantity every headline here is weighted by.
This measures what the matched set and the unplaced set are worth in the float64 gradient`s
own squared norm, per block and per leaf name, so the scope of anything scored later is a
measured share and not an assumed one.
"""
import json, sys
import torch

F64 = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
MODEL_TOTAL_SQ = 10.279642678524985
PRE = "pairformer_stack.blocks."
ABSENT_LEAVES = ["attn_pair_bias.linear_z.weight", "attn_pair_bias.mha.linear_q.weight",
                 "attn_pair_bias.mha.linear_q.bias", "attn_pair_bias.mha.linear_k.weight",
                 "attn_pair_bias.mha.linear_v.weight"]

g = torch.load(F64, map_location="cpu", weights_only=False)
total = sum(float(torch.linalg.vector_norm(v.to(torch.float64)))**2
            for v in g.values() if v is not None)
sq = {}
for k, v in g.items():
    if k.startswith(PRE) and v is not None:
        sq[k] = float(torch.linalg.vector_norm(v.to(torch.float64)))**2
stack = sum(sq.values())
absent = {k: s for k, s in sq.items() if k.split(".", 3)[3] in ABSENT_LEAVES}
matched = {k: s for k, s in sq.items() if k not in absent}
per_block = {}
for i in range(48):
    p = f"{PRE}{i}."
    b = {k: s for k, s in sq.items() if k.startswith(p)}
    ba = {k: s for k, s in absent.items() if k.startswith(p)}
    per_block[i] = {"n": len(b), "sq": sum(b.values()),
                    "pct_of_model": 100.0*sum(b.values())/total,
                    "pct_of_stack": 100.0*sum(b.values())/stack,
                    "n_unplaced": len(ba),
                    "unplaced_pct_of_block": 100.0*sum(ba.values())/sum(b.values())}
by_leaf = {}
for k, s in sq.items():
    by_leaf.setdefault(k.split(".", 3)[3], [0, 0.0])
    by_leaf[k.split(".", 3)[3]][0] += 1
    by_leaf[k.split(".", 3)[3]][1] += s
rep = {
 "what": __doc__.strip().splitlines()[0],
 "float64_reference": F64,
 "model_squared_gradient_norm_measured": total,
 "model_squared_gradient_norm_published": MODEL_TOTAL_SQ,
 "pairformer_stack": {"n": len(sq), "sq": stack, "pct_of_model": 100.0*stack/total},
 "matched_by_the_device_bijection": {
   "n": len(matched), "sq": sum(matched.values()),
   "pct_of_model": 100.0*sum(matched.values())/total,
   "pct_of_the_stack": 100.0*sum(matched.values())/stack},
 "unplaced": {"n": len(absent), "sq": sum(absent.values()),
   "pct_of_model": 100.0*sum(absent.values())/total,
   "pct_of_the_stack": 100.0*sum(absent.values())/stack,
   "leaves": ABSENT_LEAVES},
 "per_block": per_block,
 "by_leaf_name": {k: {"n": v[0], "sq": v[1], "pct_of_the_stack": 100.0*v[1]/stack}
                  for k, v in sorted(by_leaf.items(), key=lambda kv: -kv[1][1])},
}
json.dump(rep, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("pairformer_stack", "matched_by_the_device_bijection",
                                      "unplaced")}, indent=1))
top = sorted(per_block.items(), key=lambda kv: -kv[1]["pct_of_stack"])[:6]
for i, d in top:
    print("block %2d  %7.3f %% of the stack  %7.4f %% of the model  unplaced %6.3f %%"
          % (i, d["pct_of_stack"], d["pct_of_model"], d["unplaced_pct_of_block"]))
