"""of3t-ditgap: WHERE inside diffusion_transformer the 2.019x lives.

Decomposed as a DIFFERENCE OF ABSOLUTE ERRORS, sum_i ||g_ours-g_f64||^2 - ||g_up-g_f64||^2,
which is additive by construction. A share moves when its denominator collapses; a difference
does not. Every figure is at padded width 384 with 56 real tokens (diffcap043).
"""
import json, collections, math

S = "perf/of3t_modelboundary/sidecar_modelboundary"
SEC = "diffusion_module.diffusion_transformer"
PRE = SEC + ".blocks."

ours = {r["param"]: r for r in json.load(open(f"{S}/renorm_vs_FLOAT64.json"))
        if r["section"] == SEC}
up = {r["param"]: r for r in json.load(open(f"{S}/UPSTREAM_BF16_vs_FLOAT64.json"))
      if r["section"] == SEC}
assert set(ours) == set(up), (len(ours), len(up))

# --- recompute the headline from the per-tensor rows, do not quote it ------------------------
sq_ref = sum(r["ref_norm"] ** 2 for r in ours.values())
sq_o = sum(r["diff_norm"] ** 2 for r in ours.values())
sq_u = sum(r["diff_norm"] ** 2 for r in up.values())
mw_o, mw_u = math.sqrt(sq_o / sq_ref), math.sqrt(sq_u / sq_ref)
head = dict(n_tensors=len(ours),
            ours_vs_float64_recomputed=mw_o, upstream_bf16_vs_float64_recomputed=mw_u,
            ours_over_upstream_recomputed=mw_o / mw_u,
            published_ours=0.11670185505910514, published_up=0.057809234623039614,
            published_ratio=2.0187407050117585,
            abs_sq_ours=sq_o, abs_sq_upstream=sq_u, abs_sq_reference=sq_ref,
            DIFFERENCE_of_absolute_errors=sq_o - sq_u)

TOT = sq_o - sq_u

def leaf(p):            # leaf family: everything after the block index
    return p[len(PRE):].split(".", 1)[1]

def blk(p):
    return int(p[len(PRE):].split(".", 1)[0])

def subblock(p):
    return leaf(p).split(".", 1)[0]

def group(fn):
    g = collections.defaultdict(float)
    for p in ours:
        g[fn(p)] += ours[p]["diff_norm"] ** 2 - up[p]["diff_norm"] ** 2
    return dict(sorted(g.items(), key=lambda kv: -kv[1]))

def pct(g):
    return {k: dict(difference=v, pct_of_difference=100 * v / TOT) for k, v in g.items()}

by_sub, by_leaf, by_block = pct(group(subblock)), pct(group(leaf)), pct(group(blk))

# LayerNorm affine leaves in the DiT -- the trunk's signature object (D191: 93.80 %)
LN_AFFINE = {"attention_pair_bias.layer_norm_a.layer_norm_s.weight",
             "attention_pair_bias.layer_norm_z.weight",
             "conditioned_transition.layer_norm.layer_norm_s.weight"}
ADALN = {"attention_pair_bias.layer_norm_a.linear_s.weight",
         "attention_pair_bias.layer_norm_a.linear_g.weight",
         "attention_pair_bias.layer_norm_a.linear_g.bias",
         "attention_pair_bias.linear_ada_out.weight",
         "attention_pair_bias.linear_ada_out.bias",
         "conditioned_transition.layer_norm.linear_s.weight",
         "conditioned_transition.layer_norm.linear_g.weight",
         "conditioned_transition.layer_norm.linear_g.bias",
         "conditioned_transition.linear_g.weight",
         "conditioned_transition.linear_g.bias"}
MATMUL = {"attention_pair_bias.mha.linear_o.weight", "attention_pair_bias.mha.linear_g.weight",
          "attention_pair_bias.linear_z.weight", "conditioned_transition.linear_out.weight",
          "conditioned_transition.swiglu.linear_a.weight",
          "conditioned_transition.swiglu.linear_b.weight"}
fam = {"layer_norm_affine (3 leaves)": LN_AFFINE, "adaLN_gate (10 leaves)": ADALN,
       "matmul_weight (6 leaves)": MATMUL}
by_class = {}
for k, st in fam.items():
    v = sum(ours[p]["diff_norm"] ** 2 - up[p]["diff_norm"] ** 2 for p in ours if leaf(p) in st)
    by_class[k] = dict(difference=v, pct_of_difference=100 * v / TOT,
                       n_tensors=sum(1 for p in ours if leaf(p) in st))
assert sum(len(v) for v in fam.values()) == 19, sum(len(v) for v in fam.values())

# --- per tensor, the top of the difference, with the three readings the brief asks for -------
rows = []
for p in ours:
    o, u = ours[p], up[p]
    rows.append(dict(param=p, difference=o["diff_norm"] ** 2 - u["diff_norm"] ** 2,
                     pct_of_difference=100 * (o["diff_norm"] ** 2 - u["diff_norm"] ** 2) / TOT,
                     ref_norm=o["ref_norm"],
                     ours_rel_l2=o["rel_l2"], ours_norm_ratio=o["arm_norm"] / o["ref_norm"],
                     ours_cos=o["cos"],
                     up_rel_l2=u["rel_l2"], up_norm_ratio=u["arm_norm"] / u["ref_norm"],
                     up_cos=u["cos"],
                     ours_over_up_rel=o["rel_l2"] / u["rel_l2"] if u["rel_l2"] else None,
                     pct_of_model_mass=o["pct_of_model_mass"]))
rows.sort(key=lambda r: -r["difference"])

cum, conc = 0.0, None
for i, r in enumerate(rows, 1):
    cum += r["difference"]
    if conc is None and cum >= 0.6 * TOT:
        conc = dict(n_tensors_for_60pct=i, share_of_456=100 * i / len(rows))
worst_rel = max(rows, key=lambda r: r["ours_rel_l2"])

out = dict(
    what="the D187 2.019x decomposed as a difference of absolute errors, not as shares",
    frame="in-frame: both arms and both references share one float64 boundary, proven "
          "bit-exact in FRAME_f64_vs_f64.json (552/552 tensors, rel_l2 exactly 0.0)",
    padded_width=384, real_tokens=56,
    capture="/home/ttuser/of3t_softgrad/diffcap043 (no_samples=48)",
    host_of_arm="tt-quietbox2 (qb2) card 3, of3t-f64softmax renorm arm",
    host_of_floor="the upstream bf16 floor is a full-model run; D189 makes it host-dependent "
                  "at the percent level and MODEL_withtrunk_n384.json records host tt-quietbox2",
    scoring_host="tt-quietbox2 (qb2), CPU only, no card, this row",
    headline=head, concentration=conc,
    by_subblock=by_sub, by_leaf_class=by_class, by_leaf_family=by_leaf, by_block=by_block,
    top25_tensors=rows[:25],
    worst_rel_l2_tensor=worst_rel,
    n_negative_difference=sum(1 for r in rows if r["difference"] < 0),
    negative_difference_total=sum(r["difference"] for r in rows if r["difference"] < 0),
)
json.dump(out, open("perf/of3t_ditgap/WHERE.json", "w"), indent=1)

print("== HEADLINE (recomputed) ==")
for k, v in head.items(): print(f"  {k} = {v}")
print("\n== CONCENTRATION ==", conc)
print(f"  tensors with a NEGATIVE difference (we beat upstream): {out['n_negative_difference']}"
      f" of {len(rows)}, total {out['negative_difference_total']:.6e}")
print("\n== BY SUB-BLOCK ==")
for k, v in by_sub.items(): print(f"  {v['pct_of_difference']:8.4f} %  {k}")
print("\n== BY LEAF CLASS ==")
for k, v in by_class.items():
    print(f"  {v['pct_of_difference']:8.4f} %  {k}  n={v['n_tensors']}")
print("\n== BY LEAF FAMILY (all 19) ==")
for k, v in by_leaf.items(): print(f"  {v['pct_of_difference']:8.4f} %  {k}")
print("\n== BY BLOCK ==")
for k, v in by_block.items(): print(f"  block {k:2d}: {v['pct_of_difference']:8.4f} %")
print("\n== TOP 12 TENSORS ==")
for r in rows[:12]:
    print(f"  {r['pct_of_difference']:7.3f} %  {r['param'][len(PRE):]}\n"
          f"          ours rel_l2 {r['ours_rel_l2']:.6f} norm_ratio {r['ours_norm_ratio']:.6f} "
          f"cos {r['ours_cos']:.6f}\n"
          f"          up   rel_l2 {r['up_rel_l2']:.6f} norm_ratio {r['up_norm_ratio']:.6f} "
          f"cos {r['up_cos']:.6f}   ref_norm {r['ref_norm']:.6e}")
print("\n== WORST rel_l2 IN THE SECTION ==")
w = worst_rel
print(f"  {w['param']}\n   ours rel_l2 {w['ours_rel_l2']:.6e} norm_ratio {w['ours_norm_ratio']:.6f}"
      f" cos {w['ours_cos']:.6f}  ({w['pct_of_difference']:.4f} % of the difference)")
