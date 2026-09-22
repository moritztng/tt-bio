"""of3t-ditgap: is the DiT's 2.019x the trunk's object, and is it a floor or a wrong computation?

Also closes two denominator questions the decomposition exposed:
  * the sidecar scores 456 of the bundle's 552 diffusion_transformer tensors -- the 96 absent are
    the fused QKV leaves, and D187's 43.622 % mass share is computed on the 456.
  * A26's form is a reading against upstream's OWN bf16, which the section table never gave.
"""
import json, math, collections, torch

S = "perf/of3t_modelboundary/sidecar_modelboundary"
SEC = "diffusion_module.diffusion_transformer"
PRE = SEC + ".blocks."
PER_TENSOR_BAR = 0.05                       # MODEL_withtrunk_n384.json bars.per_tensor_bar
A26_MODEL_FLOOR = 0.10592054683439786
A26_MODEL_BAR = 0.14735268326440318
A26_SHAPE = A26_MODEL_BAR / A26_MODEL_FLOOR  # how A26 turns a float64 floor into a bf16 bar

o64 = {r["param"]: r for r in json.load(open(f"{S}/renorm_vs_FLOAT64.json")) if r["section"] == SEC}
u64 = {r["param"]: r for r in json.load(open(f"{S}/UPSTREAM_BF16_vs_FLOAT64.json")) if r["section"] == SEC}
obf = {r["param"]: r for r in json.load(open(f"{S}/renorm_vs_UPSTREAM_BF16.json")) if r["section"] == SEC}

def leaf(p): return p[len(PRE):].split(".", 1)[1]
def blk(p):  return int(p[len(PRE):].split(".", 1)[0])
LN = {"attention_pair_bias.layer_norm_a.layer_norm_s.weight",
      "attention_pair_bias.layer_norm_z.weight",
      "conditioned_transition.layer_norm.layer_norm_s.weight"}

def mw(d, keys):
    sr = sum(d[k]["ref_norm"] ** 2 for k in keys)
    sd = sum(d[k]["diff_norm"] ** 2 for k in keys)
    return math.sqrt(sd / sr) if sr else None

allk = list(o64)
lnk = [p for p in allk if leaf(p) in LN]
top2 = [p for p in allk if leaf(p) in
        {"attention_pair_bias.layer_norm_a.layer_norm_s.weight",
         "conditioned_transition.layer_norm.layer_norm_s.weight"}]

# ---- A26's own form: our arm against upstream's own bf16, with the section-local bar ---------
a26 = {}
for name, keys in [("section (456 tensors)", allk), ("LayerNorm affine (72)", lnk),
                   ("the two carrying leaves (48)", top2)]:
    floor = mw(u64, keys)
    a26[name] = dict(n=len(keys), ours_vs_float64=mw(o64, keys),
                     upstream_bf16_vs_float64_FLOOR=floor,
                     ours_over_floor=mw(o64, keys) / floor,
                     ours_vs_upstream_bf16_A26_FORM=mw(obf, keys),
                     A26_bar_local=A26_SHAPE * floor,
                     A26_multiple=mw(obf, keys) / (A26_SHAPE * floor))

# ---- floor or wrong computation: does upstream's OWN bf16 also miss the per-tensor bar? -----
def over(d, keys): return sum(1 for k in keys if d[k]["rel_l2"] > PER_TENSOR_BAR)
floor_ev = {}
for name, keys in [("all 456", allk), ("LayerNorm affine 72", lnk), ("two carrying leaves 48", top2)]:
    floor_ev[name] = dict(
        n=len(keys), ours_over_0p05=over(o64, keys), upstream_own_bf16_over_0p05=over(u64, keys),
        ours_worst_cos=min(o64[k]["cos"] for k in keys),
        upstream_worst_cos=min(u64[k]["cos"] for k in keys),
        ours_norm_ratio_range=[min(o64[k]["arm_norm"] / o64[k]["ref_norm"] for k in keys),
                               max(o64[k]["arm_norm"] / o64[k]["ref_norm"] for k in keys)],
        upstream_norm_ratio_range=[min(u64[k]["arm_norm"] / u64[k]["ref_norm"] for k in keys),
                                   max(u64[k]["arm_norm"] / u64[k]["ref_norm"] for k in keys)])

# ---- per-block reference mass on the carriers: are deep blocks clean, or empty? --------------
bm = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for p in top2:
    b = blk(p)
    bm[b][0] += o64[p]["ref_norm"] ** 2
    bm[b][1] += o64[p]["diff_norm"] ** 2
    bm[b][2] += u64[p]["diff_norm"] ** 2
tot_ref = sum(v[0] for v in bm.values())
byblock = {b: dict(pct_of_carrier_reference_mass=100 * v[0] / tot_ref,
                   ours_rel_l2=math.sqrt(v[1] / v[0]), upstream_rel_l2=math.sqrt(v[2] / v[0]),
                   ours_over_upstream=math.sqrt(v[1] / v[2]))
           for b, v in sorted(bm.items())}

# ---- the 96 tensors the sidecar does not score ----------------------------------------------
mdl = torch.load("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt",
                 map_location="cpu", weights_only=False)
bundle_dit = [k for k in mdl if k.startswith(SEC + ".")]
absent = sorted(set(bundle_dit) - set(allk))
sq = lambda ks: sum(float(mdl[k].double().pow(2).sum()) for k in ks)
tot_model_sq = sum(float(v.double().pow(2).sum()) for v in mdl.values() if v is not None)
cover = dict(
    bundle_dit_tensors=len(bundle_dit), scored_by_sidecar=len(allk), absent=len(absent),
    absent_leaf_families=sorted({leaf(k) for k in absent}),
    absent_pct_of_model_squared_norm=100 * sq(absent) / tot_model_sq,
    scored_pct_of_model_squared_norm=100 * sq(allk) / tot_model_sq,
    dit_true_pct_of_model_squared_norm=100 * sq(bundle_dit) / tot_model_sq,
    published_pct_of_model_mass=43.622,
    why="our AttentionPairBias fuses Q, K and V into one projection, so the four unfused leaves "
        "carry no separately-named device gradient and the sidecar has nothing to pair them with")

# ---- the trunk's signature beside this one ---------------------------------------------------
sig = dict(
    trunk=dict(source="D191 UPDATE pass 342, perf/of3t_widthattr/GROWTH.json, qb1",
               quantity="excess of our width-384 arm over width-64, mw^2 units",
               layer_norm_affine_share_pct=93.80, n_families=24,
               top_families={"attn_pair_bias.layer_norm_a.weight": 34.89,
                             "attn_pair_bias.layer_norm_a.bias": 27.43,
                             "single_transition.layer_norm.bias": 10.22,
                             "single_transition.layer_norm.weight": 7.39},
               concentration="83.85 % in 20 of 2,736 tensors; 3 of 48 blocks carry 74.58 %",
               worst_tensor="blocks.4.attn_pair_bias.layer_norm_a.weight",
               worst_norm_ratio=7.3729, worst_cos=-0.0057,
               shape="norm_ratio 7.37 at cos -0.0057 -- ORTHOGONAL to the reference, a wrong "
                     "computation, reported at a leaf of3t-lnaffine found innocent"),
    diffusion_transformer=dict(source="this row, perf/of3t_ditgap/WHERE.json, qb2",
                               quantity="difference of absolute errors ours minus upstream bf16, "
                                        "both against float64, at padded width 384",
                               layer_norm_affine_share_pct=97.7612, n_families=3,
                               top_families={
                                   "attention_pair_bias.layer_norm_a.layer_norm_s.weight": 61.2093,
                                   "conditioned_transition.layer_norm.layer_norm_s.weight": 36.5721},
                               concentration="60 % in 6 of 456 tensors; 4 of 24 blocks "
                                             "carry 61.65 %",
                               worst_tensor="blocks.17.attention_pair_bias.layer_norm_a."
                                            "layer_norm_s.weight",
                               worst_rel_l2=3.227821e-01, worst_norm_ratio=0.800875,
                               worst_cos=0.959708,
                               shape="norm_ratio 0.80 to 1.20 at cos 0.96 to 0.999 -- ALIGNED "
                                     "with the reference, a magnitude error in a "
                                     "cancellation-limited reduction, not a wrong computation"))

out = dict(what="D187's signature, its A26-form reading, and the denominator the section table used",
           padded_width=384, real_tokens=56,
           scoring_host="tt-quietbox2 (qb2), CPU only, no card",
           A26_shape_floor_to_bar=A26_SHAPE, a26=a26, floor_evidence=floor_ev,
           carriers_by_block=byblock, coverage=cover, signature=sig)
json.dump(out, open("perf/of3t_ditgap/SIGNATURE.json", "w"), indent=1)

print("== A26 FORM (in-frame, section-local floors) ==")
for k, v in a26.items():
    print(f"  {k}: ours/f64 {v['ours_vs_float64']:.6f}  floor {v['upstream_bf16_vs_float64_FLOOR']:.6f}"
          f"  = {v['ours_over_floor']:.4f}x floor")
    print(f"      A26 form: ours vs their bf16 {v['ours_vs_upstream_bf16_A26_FORM']:.6f} against "
          f"local bar {v['A26_bar_local']:.6f}  = {v['A26_multiple']:.4f}x A26")
print("\n== FLOOR OR WRONG COMPUTATION ==")
for k, v in floor_ev.items():
    print(f"  {k}: ours over 0.05 bar {v['ours_over_0p05']}/{v['n']}, "
          f"UPSTREAM'S OWN bf16 over it {v['upstream_own_bf16_over_0p05']}/{v['n']}")
    print(f"      cos worst ours {v['ours_worst_cos']:.6f} / upstream {v['upstream_worst_cos']:.6f}"
          f"   norm_ratio ours {v['ours_norm_ratio_range'][0]:.4f}-{v['ours_norm_ratio_range'][1]:.4f}"
          f" upstream {v['upstream_norm_ratio_range'][0]:.4f}-{v['upstream_norm_ratio_range'][1]:.4f}")
print("\n== CARRIERS BY BLOCK (48 tensors, two leaf families) ==")
for b, v in byblock.items():
    print(f"  block {b:2d}  ref mass {v['pct_of_carrier_reference_mass']:6.3f} %   "
          f"ours {v['ours_rel_l2']:.6f}  up {v['upstream_rel_l2']:.6f}  "
          f"{v['ours_over_upstream']:.4f}x")
print("\n== COVERAGE ==")
for k, v in cover.items(): print(f"  {k} = {v}")
