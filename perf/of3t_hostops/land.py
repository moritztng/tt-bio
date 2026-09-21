#!/usr/bin/env python3
"""Does a gradient actually LAND on the leaf the shipped walk registers?

`reach.py` proves the distogram weight is now in the walked set and that the optimizer can
step it: it injects a gradient and rebinds. It does not prove that `backward` puts one there,
and a lever can fire and be inert. So this discovers parameters the way the shipped training
loop does -- `walk_device_weights` then `ag.parameter`, nothing hand-registered -- runs one
taped `forward_device`, backwards it, and asks which of the sixteen head weights came back
with a gradient.

Controls, because "it has a gradient" is easy to get wrong in the direction you want:
  * BEFORE: the same walk on a head whose materialization is disabled. The distogram weight
    must be ABSENT from the walked set entirely. If it is present in both arms the walk was
    never the missing piece and this whole change is inert.
  * SEED ZERO: the same taped forward and backward with a zero cotangent. Every gradient must
    come back exactly zero. A leaf that reports a non-zero gradient under a zero seed is
    reading something other than this backward.
No card-side fold is needed and none is run: the question is where a gradient lands, not what
it is worth. What it is worth is `perf/of3t_auxgrad`'s instrument, run separately.
"""
import json, os, sys, time
import numpy as np
import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio import autograd as ag                                            # noqa: E402
from tt_bio.tenstorrent import get_device, walk_device_weights               # noqa: E402
from tt_bio.openfold3_confidence import OF3ConfidenceHead                    # noqa: E402

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
WANT = "distogram.linear.weight"
N = 64

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def ft(x, dt=ttnn.bfloat16):
    return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def build(materialize):
    # The "before" arm is the pre-change state exactly: a head whose weight cache is empty
    # when the walk runs, which is what every walk in this port has seen, because inference
    # takes the host s-path and never calls `_wd`.
    h = OF3ConfidenceHead(aux, dev, ckc)
    if not materialize:
        h._wd_cache.clear()
        h.__dict__.pop("_wd_materialized", None)
    return h


def walked(head):
    """The shipped discovery: walk, then register. Nothing else."""
    found = list(walk_device_weights(head))
    return found, {p: ag.parameter(t) for p, _o, _k, t in found}


def head_weight_paths(found):
    return sorted(p for p, _o, _k, _t in found if "_wd_cache" in p)


torch.manual_seed(0)
inputs = dict(si=torch.randn(1, N, 449), st=torch.randn(1, N, 384),
              zt=torch.randn(1, N, N, 128), oh=torch.rand(1, N, N, 39).round())

res = {"n_tokens": N}
for arm in ("before", "after"):
    head = build(arm == "after")
    found, params = walked(head)
    paths = head_weight_paths(found)
    has = [p for p in paths if WANT in p]
    res[arm] = {"walked": len(found), "head_cache_weights": len(paths),
                "distogram_present": bool(has), "distogram_path": has[0] if has else None}
    print("%-7s walked %4d device tensors, %2d of them head-cache weights, distogram %s"
          % (arm, len(found), len(paths), "PRESENT" if has else "ABSENT"), flush=True)
    if arm == "before":
        continue

    for seed_name, seed_scale in (("ones", 1.0), ("zero", 0.0)):
        for t in params.values():
            t.grad = None
        with ag.tape():
            out = head.forward_device(ag.Tensor(ft(inputs["si"])), ag.Tensor(ft(inputs["st"])),
                                      ag.Tensor(ft(inputs["zt"])), ag.Tensor(ft(inputs["oh"])))
        roots, seeds = [], []
        for k in ("distogram_logits", "pae_logits", "pde_logits", "plddt_logits",
                  "experimentally_resolved_logits"):
            o = out[k]
            roots.append(o)
            shape = tuple(getattr(o, "value", o).shape)
            seeds.append(ft(torch.full(shape, seed_scale / max(1, np.prod(shape)))))
        ag.backward(roots, seeds)
        got = {}
        for p in paths:
            g = params[p].grad
            if g is None:
                got[p] = None
            else:
                gt = torch.Tensor(ttnn.to_torch(getattr(g, "value", g))).double()
                got[p] = float(gt.abs().sum())
        with_grad = [p for p, v in got.items() if v is not None]
        nonzero = [p for p, v in got.items() if v is not None and v > 0]
        dp = res[arm]["distogram_path"]
        print("  seed %-4s  %2d of %2d head weights have a gradient, %2d of them non-zero; "
              "distogram |g|_1 = %s"
              % (seed_name, len(with_grad), len(paths), len(nonzero),
                 ("%.6e" % got[dp]) if got.get(dp) is not None else "NONE"), flush=True)
        res[arm]["seed_" + seed_name] = {
            "head_weights_with_gradient": len(with_grad),
            "head_weights_nonzero_gradient": len(nonzero),
            "distogram_abs_sum": got.get(dp),
            "per_weight_abs_sum": got}

b, a = res["before"], res["after"]
ok = (not b["distogram_present"] and a["distogram_present"]
      and a["seed_ones"]["distogram_abs_sum"] and a["seed_ones"]["distogram_abs_sum"] > 0
      and a["seed_zero"]["distogram_abs_sum"] == 0.0
      and a["seed_zero"]["head_weights_nonzero_gradient"] == 0)
res["verdict"] = "LANDS" if ok else "DOES NOT LAND"
print("\nwalked device tensors %d -> %d, head-cache weights %d -> %d"
      % (b["walked"], a["walked"], b["head_cache_weights"], a["head_cache_weights"]))
print("VERDICT: %s" % res["verdict"])
json.dump(res, open("perf/of3t_hostops/LAND.json", "w"), indent=1)
print("wrote perf/of3t_hostops/LAND.json")
