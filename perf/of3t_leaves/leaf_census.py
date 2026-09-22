#!/usr/bin/env python3
"""Leaves against total, for EACH model's own pair stack. One harness, five weight remaps.

The question this row was handed is whether Protenix, Boltz-2, BoltzGen and AF2 carry the same
discovery gap OpenFold3 does, and the brief is explicit that it is one measurement each and not
an assumption in either direction. So the per-model part is only the weight remap -- genuinely
different, per family -- and everything after the build is the same code reading the same three
numbers:

    loader      what a hook on `Module.torch_to_tt` sees, which is what was reported before;
    reachable   what a walk of the BUILT model reaches, which is the denominator;
    with_grad   how many of those carry a gradient after a real taped backward.

Ordering is load-bearing and is measured rather than asserted: the walk is taken BEFORE and
AFTER the first forward, because `TriangleMultiplication` pushes its fused in-projection on the
first call at a given chunk width and a walk before that call misses it from the TOTAL as well
as from the leaves.

    python3 perf/of3t_leaves/leaf_census.py --model of3 --blocks 4 --tokens 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

OUT = "perf/of3t_leaves"
#: (32, 4, 24, 16) is `c_hidden_pair_att`, `no_heads_pair`, `c_hidden_pair_bias`,
#: `no_heads_pair_bias`. All four pairformer families ship the same four; asserted below off the
#: weight shapes rather than trusted, because a wrong head count is a silently different module.
PF_DIMS = (32, 4, 24, 16)


def _load(path, weights_only=False):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
    for key in ("state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return sd


def _by_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _truncate(flat, blocks):
    if not blocks:
        return flat
    keep = {f"layers.{i}." for i in range(blocks)}
    return {k: v for k, v in flat.items()
            if any(k.startswith(p) for p in keep)}


# ------------------------------------------------------------------ the five weight remaps

def w_of3():
    from tt_bio.openfold3_weights import remap_pairformer_stack
    sd = _load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
    return remap_pairformer_stack(sd, prefix="pairformer_stack")


def w_protenix():
    from protenix_reference import remap_pairformer_block
    ck = _load("/home/ttuser/.boltz/protenix-v2.pt", weights_only=True)
    pre = "module.pairformer_stack.blocks."
    nb = 1 + max(int(re.search(r"blocks\.(\d+)\.", k).group(1))
                 for k in ck if k.startswith(pre))
    out = {}
    for i in range(nb):
        blk = _by_prefix(ck, f"{pre}{i}.")
        for k, v in remap_pairformer_block(blk).items():
            out[f"layers.{i}.{k}"] = v
    return out


def w_boltz2():
    return _by_prefix(_load("/home/ttuser/.boltz/boltz2_conf.ckpt"), "pairformer_module.")


def w_boltzgen():
    import torch
    from tt_bio.boltzgen import adapter
    path = "/home/ttuser/.boltz/boltzgen/boltzgen1_diverse.ckpt"
    with adapter._legacy_pickle_compat():
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = adapter._remap_legacy_state_dict_keys(ck["state_dict"])
    for pre in ("pairformer_module.", "model.pairformer_module."):
        flat = _by_prefix(state, pre)
        if flat:
            return flat
    raise SystemExit("no pairformer_module keys in the BoltzGen checkpoint: "
                     f"{sorted({k.split('.')[0] for k in state})}")


def w_af2():
    from tt_bio.af2_weights import load_af2_state_dict
    npz = sorted(glob.glob(os.path.expanduser("~/.boltz/af2/params/*ptm*.npz")))
    if not npz:
        raise SystemExit("no AF2 params_model_*_ptm.npz under ~/.boltz/af2/params")
    return load_af2_state_dict(npz[0])


REMAPS = {"of3": w_of3, "protenix": w_protenix, "boltz2": w_boltz2,
          "boltzgen": w_boltzgen, "af2": w_af2}


def build(model, blocks, ckc):
    """`(module, args, kwargs, note)`. The only per-model code past the remap."""
    import ttnn
    from tt_bio.tenstorrent import Pairformer
    flat = REMAPS[model]()
    if model == "af2":
        # AF2 has no Pairformer: its pair track is `AF2PairBlock`, the same two triangle
        # multiplications and two triangle attentions in AF2's own assembly. It is also the
        # only checkpoint in the repo whose trimul carries biases, so it is the one model that
        # exercises `_gp_bias_cache` -- a second lazily fused cache the loader cannot see.
        from tt_bio.af2 import AF2PairBlock
        from tt_bio.tenstorrent import WeightScope
        scope = WeightScope.wrap(flat)
        mods = [AF2PairBlock(scope.child(f"evoformer.{i}"), ckc)
                for i in range(blocks or 1)]

        class _Stack:
            def __init__(self, blocks):
                self.blocks = blocks

            def __call__(self, z):
                for b in self.blocks:
                    z = b(z)
                return z

        c_z = flat["evoformer.0.pair_transition.norm.weight"].shape[0]
        return _Stack(mods), ("z",), {"c_z": c_z, "c_s": None}, f"{len(mods)} AF2PairBlock"
    flat = _truncate(flat, blocks)
    nb = 1 + max(int(k.split(".")[1]) for k in flat if k.startswith("layers."))
    c_z = flat["layers.0.tri_mul_out.norm_in.weight"].shape[0]
    c_s = flat["layers.0.pre_norm_s.weight"].shape[0]
    n_heads = flat["layers.0.tri_att_start.linear.weight"].shape[0]
    head_dim = flat["layers.0.tri_att_start.mha.linear_q.weight"].shape[0] // n_heads
    apb_nh = flat["layers.0.attention.proj_z.1.weight"].shape[0]
    # Read off the weights, not asserted against one family's numbers: Protenix-v2 runs 8
    # triangle-attention heads where the other three run 4, and hard-coding OF3's four would
    # build a silently different module out of Protenix's checkpoint.
    dims = (head_dim, n_heads, c_s // apb_nh, apb_nh)
    extra = dict(scale_pair_bias=False, fp32_softmax=True) if model == "of3" else {}
    pf = Pairformer(nb, *dims, True, flat, ckc, **extra)
    return (pf, ("s", "z"), {"c_z": c_z, "c_s": c_s, "dims": dims},
            f"{nb}-block Pairformer, dims {dims}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(REMAPS))
    p.add_argument("--blocks", type=int, default=4, help="0 keeps the whole stack")
    p.add_argument("--tokens", type=int, default=64)
    a = p.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from tt_bio.train.checks import weight_coverage
    from tt_bio.train.lora import walked_weights

    t0 = time.perf_counter()
    # The loader figure, for the comparison this row exists to make: a hook on `torch_to_tt`
    # is what reported 2119, and it is recorded here on the same build as the walk.
    loaded = []
    orig = T.Module.torch_to_tt
    def recording(self, key, *ar, **kw):
        t = orig(self, key, *ar, **kw)
        loaded.append((key, t))
        return t
    T.Module.torch_to_tt = recording

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    mod, order, shapes, note = build(a.model, a.blocks, ckc)
    T.Module.torch_to_tt = orig
    n = a.tokens
    rng = torch.Generator().manual_seed(7)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    raw = {}
    if shapes["c_s"] is not None:
        raw["s"] = ft(torch.randn(1, n, shapes["c_s"], generator=rng) * 0.05)
    raw["z"] = ft(torch.randn(1, n, n, shapes["c_z"], generator=rng) * 0.05)

    before = device_weights(mod)
    report = {"model": a.model, "note": note, "tokens": n, "dims": list(shapes.get("dims", ())),
              "blocks_requested": a.blocks,
              "weights_from_loader": len(loaded),
              "reachable_before_forward": len(before)}
    print(f"[{time.perf_counter()-t0:.0f}s] {a.model}: {note}, {len(loaded)} from the loader, "
          f"{len(before)} reachable before any forward", flush=True)

    # Discovery through the SHIPPED path: one `no_grad` forward, then the walk. Not a private
    # copy of it -- if this disagrees with what a training run gets, the harness is lying.
    params = walked_weights(lambda: mod(*[raw[k] for k in order]), None, mod)
    after = device_weights(mod)
    report.update(reachable_after_forward=len(after), registered=len(params),
                  materialised_by_the_forward=sorted(set(after) - set(before))[:12],
                  n_materialised_by_the_forward=len(set(after) - set(before)))
    print(f"[{time.perf_counter()-t0:.0f}s] {len(after)} reachable after one forward "
          f"(+{len(after) - len(before)} the forward materialised), "
          f"{len(params)} registered as leaves", flush=True)

    err = None
    try:
        args = [ag.Tensor(raw[k], requires_grad=True) for k in order]
        with ag.tape():
            out = mod(*args)
        outs = [o for o in (out if isinstance(out, (list, tuple)) else [out])
                if isinstance(o, ag.Tensor)]
        ag.backward(outs)
    except Exception as e:                      # a gap is a finding, not a crash
        err = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()

    cov = weight_coverage(mod)
    report.update(taped_backward_error=err, total=cov.total, leaves=cov.registered,
                  with_grad=cov.with_grad, unregistered=cov.unregistered,
                  without_grad=cov.without_grad,
                  without_grad_tails=sorted({x.split(".")[-1] for x in cov.without_grad}))
    print(f"[{time.perf_counter()-t0:.0f}s] {cov}", flush=True)
    print(f"  LEAVES AGAINST TOTAL: {cov.registered} of {cov.total} registered, "
          f"{cov.with_grad} of {cov.total} with a gradient "
          f"({100.0 * cov.with_grad / cov.total:.1f} %)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"leaves_{a.model}_b{a.blocks}_n{n}.json")
    json.dump(report, open(path, "w"), indent=1, sort_keys=True)
    print(f"-> {path}", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
