"""of3t-ditgap: is the object per-LEAF or per-BLOCK?

CREATED.json showed the LayerNorm weight is only 1.07x-1.19x its worst cotangent-sharing sibling,
so the 97.76 % the LayerNorm affine leaves carry of the DIFFERENCE is mass, not a worse ratio.
This measures the two spreads directly: within a block across its 19 leaf families, and across
the 24 blocks. Whichever is larger is where the object lives.
"""
import json, math, statistics, collections

S = "perf/of3t_modelboundary/sidecar_modelboundary"
SEC = "diffusion_module.diffusion_transformer"
PRE = SEC + ".blocks."
o = {r["param"]: r for r in json.load(open(f"{S}/renorm_vs_FLOAT64.json")) if r["section"] == SEC}
u = {r["param"]: r for r in json.load(open(f"{S}/UPSTREAM_BF16_vs_FLOAT64.json")) if r["section"] == SEC}
def leaf(p): return p[len(PRE):].split(".", 1)[1]
def blk(p):  return int(p[len(PRE):].split(".", 1)[0])
LN = {"attention_pair_bias.layer_norm_a.layer_norm_s.weight",
      "attention_pair_bias.layer_norm_z.weight",
      "conditioned_transition.layer_norm.layer_norm_s.weight"}

sec_ref = sum(r["ref_norm"] ** 2 for r in o.values())
ln_ref = sum(o[p]["ref_norm"] ** 2 for p in o if leaf(p) in LN)
two = {"attention_pair_bias.layer_norm_a.layer_norm_s.weight",
       "conditioned_transition.layer_norm.layer_norm_s.weight"}
two_ref = sum(o[p]["ref_norm"] ** 2 for p in o if leaf(p) in two)
mass = dict(layer_norm_affine_pct_of_section_reference_mass=100 * ln_ref / sec_ref,
            two_carrying_leaves_pct_of_section_reference_mass=100 * two_ref / sec_ref,
            layer_norm_affine_pct_of_section_DIFFERENCE=97.7612,
            reading="the LayerNorm affine leaves carry the difference because they hold the "
                    "reference mass, not because their ratio is worse")

def ratio(keys):
    so = sum(o[k]["diff_norm"] ** 2 for k in keys)
    su = sum(u[k]["diff_norm"] ** 2 for k in keys)
    return math.sqrt(so / su) if su > 0 else None

# per (block, leaf) ratios, restricted to entries with real reference mass so a near-zero
# tensor's ratio cannot dominate a spread (A14's near-zero filter, applied as a mass floor)
FLOOR = 1e-4 * (sec_ref / len(o)) ** 0.5
cells = {}
for p in o:
    if o[p]["ref_norm"] < FLOOR: continue
    cells[(blk(p), leaf(p))] = ratio([p])

within = {}
for b in range(24):
    v = [r for (bb, _), r in cells.items() if bb == b and r]
    if len(v) > 4:
        within[b] = dict(n=len(v), median=statistics.median(v), lo=min(v), hi=max(v),
                         hi_over_lo=max(v) / min(v),
                         cv=statistics.stdev(v) / statistics.mean(v))
across = {}
for f in sorted({l for _, l in cells}):
    v = [r for (_, ll), r in cells.items() if ll == f and r]
    if len(v) > 4:
        across[f] = dict(n=len(v), median=statistics.median(v), lo=min(v), hi=max(v),
                         hi_over_lo=max(v) / min(v),
                         cv=statistics.stdev(v) / statistics.mean(v))

spread = dict(
    mass_floor_on_ref_norm=FLOOR, cells_kept=len(cells), cells_total=len(o),
    WITHIN_a_block_across_19_leaves=dict(
        median_cv=statistics.median(v["cv"] for v in within.values()),
        median_hi_over_lo=statistics.median(v["hi_over_lo"] for v in within.values())),
    ACROSS_24_blocks_per_leaf=dict(
        median_cv=statistics.median(v["cv"] for v in across.values()),
        median_hi_over_lo=statistics.median(v["hi_over_lo"] for v in across.values())),
    per_block=within, per_leaf_family=across)
spread["verdict"] = (
    "per-BLOCK" if spread["ACROSS_24_blocks_per_leaf"]["median_cv"]
    > spread["WITHIN_a_block_across_19_leaves"]["median_cv"] else "per-LEAF")

out = dict(what="whether D187's object is per-leaf or per-block", padded_width=384, real_tokens=56,
           scoring_host="tt-quietbox2 (qb2), CPU only, no card", mass=mass, spread=spread)
json.dump(out, open("perf/of3t_ditgap/STRUCTURE.json", "w"), indent=1)

print("== MASS ==")
for k, v in mass.items(): print(f"  {k} = {v}")
print(f"\n  mass floor on ref_norm {FLOOR:.3e}, kept {len(cells)} of {len(o)} tensors")
print("\n== SPREAD OF ours/upstream ==")
print(f"  WITHIN a block, across its leaf families: median CV "
      f"{spread['WITHIN_a_block_across_19_leaves']['median_cv']:.4f}, median hi/lo "
      f"{spread['WITHIN_a_block_across_19_leaves']['median_hi_over_lo']:.3f}")
print(f"  ACROSS the 24 blocks, per leaf family:    median CV "
      f"{spread['ACROSS_24_blocks_per_leaf']['median_cv']:.4f}, median hi/lo "
      f"{spread['ACROSS_24_blocks_per_leaf']['median_hi_over_lo']:.3f}")
print(f"  -> the object is {spread['verdict']}")
print("\n== PER BLOCK median ours/upstream over its leaves ==")
for b, v in sorted(within.items()):
    print(f"  block {b:2d}  median {v['median']:.4f}  range {v['lo']:.4f}-{v['hi']:.4f}  "
          f"n={v['n']}  cv {v['cv']:.4f}")
print("\n== PER LEAF FAMILY median ours/upstream over the 24 blocks ==")
for f, v in sorted(across.items(), key=lambda kv: -kv[1]["median"]):
    print(f"  {v['median']:.4f}  cv {v['cv']:.4f}  range {v['lo']:.4f}-{v['hi']:.4f}  {f}")
