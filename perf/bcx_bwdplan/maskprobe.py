"""One Evoformer block at an all-ones MSA mask: does the masked program compute the unmasked
program's function, forward and backward, with and without the backward levers?"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "perf/bcx_stack")
import stack as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--block", type=int, default=0)
ap.add_argument("--params", default=S.A.DEFAULT_PARAMS)
ap.add_argument("--card", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
lv, dev, ref = S.open_all(args)
n = args.n
m0, z0, wm, wz = S.inputs(ref, n, 0)
mask = dev.up(torch.ones(1, n))


def run(arm):
    lv.arm(arm)
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, 0, 1, evo_first=args.block,
                           msa_mask=mask if lv.mask else None)
    dev.sync()
    outs = (dev.down(mo.value, m0.shape), dev.down(zo.value, z0.shape))
    dev.ag.backward([mo, zo], [dev.seed(wm, mo), dev.seed(wz, zo)])
    dev.sync()
    return outs + (dev.grad(ml, m0.shape), dev.grad(zl, z0.shape))


res = {}
for arm in ["stack", "stack+mask", "bwd", "bwd+mask"]:
    run(arm)
    res[arm] = run(arm)
out = {"n": n, "block": args.block, "pairs": {}}
for arm, base in [("stack+mask", "stack"), ("bwd", "stack"), ("bwd+mask", "bwd"),
                  ("bwd+mask", "stack+mask")]:
    rec = {nm: {"eq": bool(torch.equal(x, y)), **S.A.cmp(x, y)}
           for nm, x, y in zip(("mo", "zo", "dm", "dz"), res[arm], res[base])}
    out["pairs"][f"{arm} vs {base}"] = rec
    print(arm, "vs", base, json.dumps(rec), flush=True)
if args.out:
    S.save(args.out, out)

# The float64 reference of the same block on the same inputs: which program is AF2's function?
f64 = ref["f64"]
xs = [m0.double(), z0.double()]

g64, o64 = S.A.ref_vjp(lambda a, b: S.A.ref_evo(f64, args.block, a, b), xs, [wm.double(), wz.double()])
out["vs_f64"] = {}
for arm, r in res.items():
    rec = {nm: S.A.cmp(x.double(), y.double())
           for nm, x, y in zip(("mo", "zo", "dm", "dz"), r, [t.detach() for t in o64] + list(g64))}
    out["vs_f64"][arm] = rec
    print(arm, "vs f64", json.dumps(rec), flush=True)
if args.out:
    S.save(args.out, out)
