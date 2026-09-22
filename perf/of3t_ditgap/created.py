"""of3t-ditgap: created at the leaf, or reported at a leaf and created upstream?

The trunk's object is REPORTED at LayerNorm affine leaves whose own arithmetic of3t-lnaffine found
innocent, with the error arriving in the cotangent written at the LayerNorm's INPUT. If that is
true here, the sibling leaves inside the SAME sub-block -- which consume the SAME cotangent by a
different reduction -- must be wrong too. If only the LayerNorm weight is wrong, the error is
created in its own reduction.
"""
import json, math, collections

S = "perf/of3t_modelboundary/sidecar_modelboundary"
SEC = "diffusion_module.diffusion_transformer"
PRE = SEC + ".blocks."
o = {r["param"]: r for r in json.load(open(f"{S}/renorm_vs_FLOAT64.json")) if r["section"] == SEC}
u = {r["param"]: r for r in json.load(open(f"{S}/UPSTREAM_BF16_vs_FLOAT64.json")) if r["section"] == SEC}
def leaf(p): return p[len(PRE):].split(".", 1)[1]
def blk(p):  return int(p[len(PRE):].split(".", 1)[0])

# The two sub-blocks, each as: the LayerNorm weight, and its cotangent-sharing siblings.
GROUPS = {
 "attention_pair_bias (AdaLN)": dict(
   norm="attention_pair_bias.layer_norm_a.layer_norm_s.weight",
   siblings=["attention_pair_bias.layer_norm_a.linear_s.weight",
             "attention_pair_bias.layer_norm_a.linear_g.weight",
             "attention_pair_bias.layer_norm_a.linear_g.bias",
             "attention_pair_bias.linear_ada_out.weight",
             "attention_pair_bias.mha.linear_o.weight"]),
 "conditioned_transition (AdaLN)": dict(
   norm="conditioned_transition.layer_norm.layer_norm_s.weight",
   siblings=["conditioned_transition.layer_norm.linear_s.weight",
             "conditioned_transition.layer_norm.linear_g.weight",
             "conditioned_transition.layer_norm.linear_g.bias",
             "conditioned_transition.linear_g.weight",
             "conditioned_transition.linear_out.weight",
             "conditioned_transition.swiglu.linear_a.weight",
             "conditioned_transition.swiglu.linear_b.weight"]),
}
# restrict to the blocks that carry the difference, so a low-mass block cannot dilute it
HOT = [5, 6, 10, 11, 12, 8, 1]

def mwr(fam, blocks):
    ks = [p for p in o if leaf(p) == fam and blk(p) in blocks]
    sr = sum(o[k]["ref_norm"] ** 2 for k in ks)
    return dict(n=len(ks), ours=math.sqrt(sum(o[k]["diff_norm"] ** 2 for k in ks) / sr),
                up=math.sqrt(sum(u[k]["diff_norm"] ** 2 for k in ks) / sr),
                ours_over_up=math.sqrt(sum(o[k]["diff_norm"] ** 2 for k in ks)
                                       / sum(u[k]["diff_norm"] ** 2 for k in ks)))

res = {}
for gname, g in GROUPS.items():
    n = mwr(g["norm"], HOT)
    sibs = {s: mwr(s, HOT) for s in g["siblings"]}
    res[gname] = dict(
        layer_norm_weight=dict(leaf=g["norm"], **n),
        siblings=sibs,
        verdict_ratio_norm_over_worst_sibling=n["ours_over_up"] / max(
            v["ours_over_up"] for v in sibs.values()))

out = dict(what="created at the LayerNorm weight's own reduction, or inherited in the cotangent",
           blocks_used=HOT, padded_width=384, real_tokens=56,
           scoring_host="tt-quietbox2 (qb2), CPU only, no card", groups=res)
json.dump(out, open("perf/of3t_ditgap/CREATED.json", "w"), indent=1)

for gname, g in res.items():
    print(f"== {gname}, blocks {HOT} ==")
    n = g["layer_norm_weight"]
    print(f"  LN WEIGHT  {n['leaf']}")
    print(f"             ours {n['ours']:.6f}  up {n['up']:.6f}  -> {n['ours_over_up']:.4f}x")
    for s, v in sorted(g["siblings"].items(), key=lambda kv: -kv[1]["ours_over_up"]):
        print(f"  sibling    {s}\n             ours {v['ours']:.6f}  up {v['up']:.6f}  "
              f"-> {v['ours_over_up']:.4f}x")
    print(f"  LN weight is {g['verdict_ratio_norm_over_worst_sibling']:.4f}x its WORST sibling\n")
