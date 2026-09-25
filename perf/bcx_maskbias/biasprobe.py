"""Which mask bias makes one Evoformer block leave the unmasked function at an all-ones mask?

Same block, same inputs and the same float64 reference as `perf/bcx_bwdplan/maskprobe.py`. Arms
swap what `AF2EvoformerBlock._mask_biases` hands the two attentions; the OPM takes the mask in
every masked arm, and it writes z only, so `mo` isolates the attentions.
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "perf/bcx_stack")
import stack as S  # noqa: E402
from tt_bio import af2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--block", type=int, default=0)
ap.add_argument("--params", default=S.A.DEFAULT_PARAMS)
ap.add_argument("--card", type=int, default=0)
ap.add_argument("--arms", default="nomask,mask,row,col,none")
ap.add_argument("--bwd", action="store_true")
ap.add_argument("--pair", action="store_true",
                help="every masked arm also forces the pair masks, as steptime's tree+ones does")
ap.add_argument("--out", default=None)
args = ap.parse_args()
lv, dev, ref = S.open_all(args)
n = args.n
m0, z0, wm, wz = S.inputs(ref, n, 0)
mask = dev.up(torch.ones(1, n))
up = lambda t: dev.ttnn.from_torch(t.to(torch.bfloat16), layout=dev.ttnn.TILE_LAYOUT,
                                   device=dev.device, dtype=dev.ttnn.bfloat16)
pm = (up(torch.ones(1, n, n)), up(torch.zeros(1, 1, 1, n))) if args.pair else (None, None)
built = af2.AF2EvoformerBlock._mask_biases
KEEP = {"mask": (1, 1), "row": (1, 0), "col": (0, 1), "none": (0, 0)}
# `pair`: the pair masks alone, MSA mask None, which isolates the pair track's masked sites.


def run(arm):
    lv.arm("stack")
    keep = KEEP.get(arm)
    if keep is not None:
        af2.AF2EvoformerBlock._mask_biases = (
            lambda self, m: tuple(b if k else None for b, k in zip(built(self, m), keep)))
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, 0, 1, evo_first=args.block,
                           msa_mask=None if keep is None else mask,
                           pair_masks=(None, None) if arm == "nomask" else pm)
    dev.sync()
    outs = (dev.down(mo.value, m0.shape), dev.down(zo.value, z0.shape))
    if args.bwd:
        dev.ag.backward([mo, zo], [dev.seed(wm, mo), dev.seed(wz, zo)])
        dev.sync()
        outs += (dev.grad(ml, m0.shape), dev.grad(zl, z0.shape))
    af2.AF2EvoformerBlock._mask_biases = built
    return outs


arms = args.arms.split(",")
res = {}
for arm in arms:
    run(arm)
    res[arm] = run(arm)
f64 = ref["f64"]
xs = [m0.double(), z0.double()]
if args.bwd:
    g64, o64 = S.A.ref_vjp(lambda a, b: S.A.ref_evo(f64, args.block, a, b), xs,
                           [wm.double(), wz.double()])
    want = [t.detach() for t in o64] + list(g64)
else:
    with torch.no_grad():
        want = list(S.A.ref_evo(f64, args.block, *xs))
names = ("mo", "zo", "dm", "dz")
out = {"n": n, "block": args.block, "stamp": S.stamp(args), "vs_f64": {}, "vs_nomask": {}}
for arm, r in res.items():
    out["vs_f64"][arm] = {nm: S.A.cmp(x.double(), y.double()) for nm, x, y in zip(names, r, want)}
    out["vs_nomask"][arm] = {nm: {"eq": bool(torch.equal(x, y)), **S.A.cmp(x.double(), y.double())}
                             for nm, x, y in zip(names, r, res[arms[0]])}
    print(arm, "f64", json.dumps({k: round(v["rel_l2"], 5) for k, v in out["vs_f64"][arm].items()}),
          "nomask", json.dumps({k: (v["eq"], round(v["rel_l2"], 5)) for k, v in out["vs_nomask"][arm].items()}),
          flush=True)
if args.out:
    (S.ROOT / "perf" / "bcx_maskbias").mkdir(exist_ok=True)
    (S.ROOT / "perf" / "bcx_maskbias" / args.out).write_text(json.dumps(out, indent=1, default=str))
