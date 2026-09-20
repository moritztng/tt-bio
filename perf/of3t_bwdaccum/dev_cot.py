#!/usr/bin/env python3
"""Our trunk's backward, instrumented: the cotangent at every block boundary, the LayerNorm
affine operands in situ, and the precision levers on that one backward formula.

This does NOT rebuild `of3t-trunkg043/dev_grad.py`. It imports it and runs its `main()` with
three patches installed, so the arm, the config spy, the fused-five placement and the whole
scoring path stay exactly the ones that produced 9.025172e+00:

  1. `ops.checkpoint_segment` is wrapped to keep a handle on each block's input and output
     tensors, and `autograd._retire` is wrapped to snapshot their gradients before the tape
     releases them. Nothing about the arithmetic changes -- `_retire` still retires.
  2. `_taped_layer_norm` is replaced by a copy of itself. With `--lever none` it is the same
     ops in the same order; every lever is one named change to it, so a reading can be
     attributed to the change and not to the rewrite.
  3. Optionally the LayerNorm backward's real operands (`x`, `g`, `gamma`) and its own dW/db
     are written out at chosen blocks, which is what lets deliverable 2 score the op in
     isolation on operands the stack actually produced.

The levers, each named for what it does to `dW = sum_t g_t * xhat_t`:

  none        production
  prod_fp32   form the summands in fp32 (D56: the reduction is precise, the 24,576 summands
              feeding it are built and rounded to bf16 first, and summing bf16 numbers in fp32
              does not recover the bits lost making them)
  sum_fp32    ask `ttnn.sum` for an fp32 OUTPUT, not just fp32 destination accumulation
              (`_taped_linear`'s own dW rule documents why: `packer_l1_acc` accumulates the
              per-K-block partials at the output dtype. `_sum_leading` passes the precise
              config and no dtype)
  xhat_fp32   recompute mean/rstd/xhat in fp32 from x
  dx_fp32     the ACTIVATION gradient path (dnorm, its two means, dx) in fp32 -- D55's four
              withheld configs, the half that propagates rather than the half that lands
  all         every one of the above
  lofi        the control: LoFi, fp32_dest_acc_en off, on the same reductions. It must make
              the reading WORSE or no flag reached the kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))

LEVERS = ("none", "prod_fp32", "sum_fp32", "xhat_fp32", "dx_fp32", "all", "lofi",
          "softmax_fp32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", default="none", choices=LEVERS)
    ap.add_argument("--cot-out", default="", help="per-rung cotangent tensors")
    ap.add_argument("--ln-capture", default="", help="comma-separated block indices")
    ap.add_argument("--ln-out", default="")
    ap.add_argument("--inject-ref", default="",
                    help="TEACHER FORCING. Before each block's backward fires, replace the "
                         "cotangent that arrived with the REFERENCE's at that rung. Every "
                         "block then computes its backward on the right input, so the "
                         "cotangent this run writes at rung k is block k's OWN injection and "
                         "not the accumulation of everything below it. One run gives all 48.")
    a, rest = ap.parse_known_args()
    a.passthrough = [x for x in rest if x != "--"]
    t0 = time.perf_counter()

    import torch
    import ttnn
    import tt_bio.ops as ops
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T

    cap_blocks = {int(x) for x in a.ln_capture.split(",") if x.strip() != ""}
    REF = None
    if a.inject_ref:
        REF = torch.load(a.inject_ref, map_location="cpu", weights_only=False)["cot"]

    # ---- 1. the block boundaries -------------------------------------------------------
    BOUND, REG, GRAD = [], {}, {}
    _real_cs = ops.checkpoint_segment

    TAPED = []

    def _cs(fn, *inputs):
        out = _real_cs(fn, *inputs)
        k = len(BOUND)
        BOUND.append({"in": list(inputs), "out": list(out)})
        for nm, ts in (("in", inputs), ("out", out)):
            for j, t in enumerate(ts):
                if isinstance(t, ag.Tensor):
                    REG[id(t)] = (k, nm, "sz"[j])
        if any(isinstance(t, ag.Tensor) for t in inputs):
            blk = len(TAPED)
            TAPED.append(k)
            if REF is not None:
                from tt_bio.tenstorrent import get_device
                dev = get_device()
                rung = blk + 1
                # rung 48 is the stack's output, where the reference's quantity includes the
                # within-block s<-z path our checkpoint boundary cannot see. Block 47 keeps
                # the real seed, which is correct to its own bf16 rounding anyway.
                for j, t in enumerate(out):
                    if not isinstance(t, ag.Tensor) or t.node is None or rung > 47:
                        continue
                    r = REF[rung]["ds" if j == 0 else "dz"]
                    if r is None:
                        continue
                    orig = t.node.fn

                    def fn(g, orig=orig, r=r, t=t):
                        gg = ttnn.from_torch(r.to(torch.float32).reshape(
                            [int(d) for d in t.value.shape]), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=t.value.dtype)
                        return orig(gg)

                    t.node.fn = fn
        return out

    ops.checkpoint_segment = _cs

    _real_retire = ag._retire

    def _retire(t):
        key = REG.get(id(t))
        if key is not None and t.grad is not None:
            GRAD[key] = ttnn.to_torch(t.grad).to(torch.float64).clone()
        return _real_retire(t)

    ag._retire = _retire

    # ---- 2. the module handle, so a captured gamma can be named ------------------------
    MOD = []
    _RealPF = T.Pairformer

    class _PF(_RealPF):
        def __init__(self, *ar, **kw):
            super().__init__(*ar, **kw)
            MOD.append(self)

    T.Pairformer = _PF
    WPATH = {}

    def _name_of(v):
        if not WPATH and MOD:
            from tt_bio.tenstorrent import device_weights
            for p, w in device_weights(MOD[0]).items():
                try:
                    WPATH[w.buffer_address()] = p
                except Exception:
                    pass
        try:
            return WPATH.get(v.buffer_address())
        except Exception:
            return None

    # ---- 3. the LayerNorm backward, with the levers -------------------------------------
    lev = a.lever
    on = (lambda n: lev == "all" or lev == n)
    CAPTURED = []

    def _precise():
        return ag.precise_config()

    def _lofi():
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
            fp32_dest_acc_en=False, packer_l1_acc=False)

    def _sum_leading(t, out_shape, fp32_out):
        flat = ag._flat2d(t)
        cfg = _lofi() if lev == "lofi" else _precise()
        if fp32_out:
            try:
                summed = ttnn.sum(flat, dim=0, keepdim=True, compute_kernel_config=cfg,
                                  dtype=ttnn.float32)
            except TypeError:
                # no dtype on this build: give it an fp32 INPUT instead, which is the same
                # request -- `ttnn.sum` keeps the operand's dtype on the way out.
                summed = ttnn.sum(ttnn.typecast(flat, ttnn.float32), dim=0, keepdim=True,
                                  compute_kernel_config=cfg)
        else:
            summed = ttnn.sum(flat, dim=0, keepdim=True, compute_kernel_config=cfg)
        return ttnn.reshape(summed, [int(d) for d in out_shape])

    def _taped_layer_norm(shipped, args, kwargs):
        """`ag._taped_layer_norm`, copied, with the levers named above and nothing else."""
        args = list(args) + [None] * (3 - len(args))
        x, gamma, beta = (ag._wrap(args[0]), ag._wrap(args[1]), ag._wrap(args[2]))
        kw = dict(kwargs)
        for nm, slot in (("weight", 1), ("bias", 2)):
            if kw.get(nm) is not None:
                v = ag._wrap(kw.pop(nm))
                gamma, beta = (v, beta) if slot == 1 else (gamma, v)
            else:
                kw.pop(nm, None)
        kw.pop("l1_headroom", None)
        eps = kw.pop("epsilon", 1e-5)
        cfg = kw.pop("compute_kernel_config", None)
        xv = x.value
        out_v = shipped(xv, weight=(gamma.value if gamma is not None else None),
                        bias=(beta.value if beta is not None else None),
                        epsilon=eps, compute_kernel_config=cfg, **kw)
        bwcfg = _lofi() if lev == "lofi" else _precise()
        parents = [t for t in (x, gamma, beta) if t is not None]

        def make():
            def bw(g):
                xv = x.value
                if on("xhat_fp32"):
                    x32 = ttnn.typecast(xv, ttnn.float32)
                    mean = ttnn.mean(x32, dim=-1, keepdim=True, compute_kernel_config=bwcfg)
                    centered = ttnn.subtract(x32, mean)
                    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                                    compute_kernel_config=bwcfg)
                    rstd = ttnn.rsqrt(ttnn.add(var, eps))
                    norm = ttnn.multiply(centered, rstd)
                else:
                    mean = ttnn.mean(xv, dim=-1, keepdim=True)
                    centered = ttnn.subtract(xv, mean)
                    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                                    compute_kernel_config=bwcfg)
                    rstd = ttnn.rsqrt(ttnn.add(var, eps))
                    norm = ttnn.multiply(centered, rstd)
                path = _name_of(gamma.value) if gamma is not None else None
                want = (path is not None
                        and int(path.split(".")[1]) in cap_blocks) if path else False
                dw = db = None
                if gamma is not None and gamma.requires_grad:
                    if on("prod_fp32"):
                        prod = ttnn.multiply(ttnn.typecast(g, ttnn.float32),
                                             ttnn.typecast(norm, ttnn.float32))
                    else:
                        prod = ttnn.multiply(g, norm)
                    dw = _sum_leading(prod, gamma.value.shape,
                                      on("sum_fp32") or on("prod_fp32"))
                    gamma.add_grad(dw)
                if beta is not None and beta.requires_grad:
                    db = _sum_leading(g, beta.value.shape, on("sum_fp32"))
                    beta.add_grad(db)
                if want:
                    CAPTURED.append({
                        "gamma_path": path,
                        "x": ttnn.to_torch(xv).to(torch.float32).clone(),
                        "g": ttnn.to_torch(g).to(torch.float32).clone(),
                        "gamma": (None if gamma is None
                                  else ttnn.to_torch(gamma.value).to(torch.float32).clone()),
                        "eps": eps,
                        "dW_device": (None if dw is None
                                      else ttnn.to_torch(dw).to(torch.float64).clone()),
                        "db_device": (None if db is None
                                      else ttnn.to_torch(db).to(torch.float64).clone()),
                        "xhat_device": ttnn.to_torch(norm).to(torch.float32).clone(),
                        "x_dtype": str(xv.dtype), "g_dtype": str(g.dtype)})
                if x.requires_grad:
                    if on("dx_fp32"):
                        g32 = ttnn.typecast(g, ttnn.float32)
                        n32 = (norm if norm.dtype == ttnn.float32
                               else ttnn.typecast(norm, ttnn.float32))
                        dnorm = (ttnn.multiply(g32, ttnn.typecast(gamma.value, ttnn.float32))
                                 if gamma is not None else g32)
                        dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True,
                                            compute_kernel_config=bwcfg)
                        dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, n32), dim=-1,
                                                 keepdim=True, compute_kernel_config=bwcfg)
                        dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                           ttnn.multiply(n32, dn_norm_mean))
                        r32 = (rstd if rstd.dtype == ttnn.float32
                               else ttnn.typecast(rstd, ttnn.float32))
                        x.add_grad(ttnn.typecast(ttnn.multiply(dx, r32), x.value.dtype))
                    else:
                        dnorm = (ttnn.multiply(g, gamma.value) if gamma is not None else g)
                        dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
                        dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1,
                                                 keepdim=True)
                        dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                           ttnn.multiply(norm, dn_norm_mean))
                        x.add_grad(ttnn.multiply(dx, rstd))
            return bw

        return ag._tape(out_v, parents, make)

    # The softmax backward: `inner = sum(g*y)` then `y*(g - inner)`, a near-cancellation whose
    # reduction is the one D55 names as carrying no kernel config and no dtype. The lever does
    # it in fp32 and counts its own firings, so "inert" can be told from "never ran".
    SM = {"fired": 0}
    _real_sm = tt._VERBS["softmax"]

    def _v_softmax_fp32(shipped, args, kwargs):
        x = ag._wrap(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        ra, rk = tt._raw(args, kwargs) if hasattr(tt, "_raw") else ag._raw(args, kwargs)
        y0 = ttnn.softmax(*ra, **rk) if shipped is ttnn.softmax_in_place else shipped(*ra, **rk)
        box = [y0]

        def make():
            def bw(g):
                SM["fired"] += 1
                y = box[0]
                g32 = ttnn.typecast(g, ttnn.float32)
                y32 = ttnn.typecast(y, ttnn.float32)
                inner = ttnn.sum(ttnn.multiply(g32, y32), dim=dim, keepdim=True,
                                 compute_kernel_config=_precise())
                x.add_grad(ttnn.typecast(
                    ttnn.multiply(y32, ttnn.subtract(g32, inner)), x.value.dtype))
            return bw

        out = ag._tape(y0, [x], make)
        if out.node is not None:
            out.box = box
        return out

    if lev == "softmax_fp32":
        for v in ("softmax", "softmax_in_place"):
            if v in tt._VERBS:
                tt._VERBS[v] = _v_softmax_fp32

    ag._taped_layer_norm = _taped_layer_norm
    ag._TAPED["layer_norm"] = _taped_layer_norm
    tt._VERBS["layer_norm"] = _taped_layer_norm

    # ---- run of3t-trunkg043's own harness ----------------------------------------------
    import dev_grad
    argv = list(a.passthrough)
    if argv and argv[0] == "--":
        argv = argv[1:]
    sys.argv = ["dev_grad.py"] + argv
    rc = dev_grad.main()

    # ---- what the instrumentation saw ---------------------------------------------------
    if a.cot_out:
        for k, ent in enumerate(BOUND):
            for nm in ("in", "out"):
                for j, t in enumerate(ent[nm]):
                    key = (k, nm, "sz"[j])
                    if key not in GRAD and isinstance(t, ag.Tensor) and t.grad is not None:
                        GRAD[key] = ttnn.to_torch(t.grad).to(torch.float64).clone()
        # `dev_grad` runs a DISCOVERY forward before the taped one, so `checkpoint_segment`
        # fires 96 times for a 48-block stack and only the second half carries taped tensors.
        # Keep the segments that actually taped; a scan indexed off the raw call count would
        # report 48 empty rungs and read as a harness that saw nothing.
        taped = list(TAPED)
        n = len(taped)
        cot = {}
        for k in range(n + 1):
            src = (taped[k], "in") if k < n else (taped[n - 1], "out")
            cot[k] = {"ds": GRAD.get((src[0], src[1], "s")),
                      "dz": GRAD.get((src[0], src[1], "z"))}
        torch.save({"cot": cot, "lever": lev, "blocks": n,
                    "inject_ref": a.inject_ref,
                    "segments_seen": len(BOUND), "segments_taped": n}, a.cot_out)
        print(json.dumps({"cot_out": a.cot_out, "rungs": len(cot),
                          "with_ds": sum(1 for v in cot.values() if v["ds"] is not None),
                          "with_dz": sum(1 for v in cot.values() if v["dz"] is not None)}))
    if a.ln_out and CAPTURED:
        torch.save({"sites": CAPTURED, "lever": lev, "blocks_captured": sorted(cap_blocks)},
                   a.ln_out)
        print(json.dumps({"ln_out": a.ln_out, "sites": len(CAPTURED),
                          "paths": sorted({c["gamma_path"] for c in CAPTURED})[:8]}))
    print(json.dumps({"lever": lev, "softmax_backward_firings": SM["fired"],
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
