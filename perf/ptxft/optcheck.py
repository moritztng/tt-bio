#!/usr/bin/env python3
"""Phase 2 controls: the moreh-vs-host optimizer decision, and the update-magnitude gate.

Three arms, and the second is the one the brief calls non-negotiable.

--moreh     does ttnn.moreh_adamw at our pin compute the same update as the host step,
            and what does each cost on an adapter-sized parameter set? Decided on
            measurement rather than taste, per the brief.
--magnitude the update-magnitude control. `hallgrad-build` measured only 0.209 of an
            eps=0.01 step surviving a bf16 round-trip on a tensor of order 1, which is
            the difference between fine-tuning and an expensive no-op. The same question
            has to be answered for the step THIS optimizer actually takes, at the
            learning rate it actually uses, on weights of the magnitude the checkpoint
            actually holds.
--roundtrip save the adapter, reload it in THIS process, and also print what a fresh
            process must reproduce. The cross-process half is `--verify`.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch


def adapter_params(device, cfg, dtype, rng):
    """One pairformer block's worth of LoRA factors, at the real Protenix v2 shapes.

    34 adapted linears per block, from perf/ptxft/param_census.py: 12 trimul (256,256),
    10 triangle-attention mha (256,256), 2 triangle-attention bias (256,8), 3 pair
    transition, 5 attention-pair-bias (384,384), 1 pair-bias z (256,16), 3 single
    transition. Sized so the cost numbers below are the cost of a real block, not of a
    toy.
    """
    from tt_bio import train as ft
    sites = ([("trimul_%d" % i, 256, 256) for i in range(12)]
             + [("triatt_mha_%d" % i, 256, 256) for i in range(10)]
             + [("triatt_bias_%d" % i, 256, 8) for i in range(2)]
             + [("pair_tr_a", 256, 1024), ("pair_tr_b", 256, 1024), ("pair_tr_o", 1024, 256)]
             + [("apb_%d" % i, 384, 384) for i in range(5)]
             + [("apb_z", 256, 16)]
             + [("single_tr_a", 384, 1536), ("single_tr_b", 384, 1536),
                ("single_tr_o", 1536, 384)])
    params = {}
    for name, fin, fout in sites:
        a, b = ft.lora_factors(fin, fout, cfg, device, dtype=dtype, rng=rng)
        params[name + ".lora_A"] = a
        params[name + ".lora_B"] = b
    return params, sites


def fake_grads(params, rng, device, dtype, scale=1.0):
    """Gradients of a plausible magnitude, so the magnitude control is not measuring zero."""
    from tt_bio import train as ft
    for t in params.values():
        shp = tuple(int(d) for d in t.value.shape)
        g = (rng.standard_normal(shp) * scale / math.sqrt(shp[0])).astype(np.float32)
        t.grad = ft.to_device(g, device, dtype=dtype)


def arm_moreh(device, dtype, args):
    import ttnn
    from tt_bio import train as ft
    print("# --moreh: does ttnn.moreh_adamw agree with the host step, and what does each cost?")
    print(f"# signature: {(ttnn.moreh_adamw.__doc__ or '').strip().splitlines()[1:8]}")
    cfg = ft.LoraConfig(rank=args.rank)
    rng = np.random.default_rng(3)
    params, _ = adapter_params(device, cfg, dtype, rng)
    n_el = sum(int(np.prod([int(d) for d in t.value.shape])) for t in params.values())
    print(f"# {len(params)} adapter tensors, {n_el:,} elements "
          f"({n_el*4/1e6:.2f} MB fp32) -- one pairformer block at rank {args.rank}")

    # --- the host step, timed
    opt = ft.AdamW(params, lr=args.lr)
    fake_grads(params, rng, device, dtype)
    opt.step()                      # warm: first call pays the torch/PCIe warmup
    ts = []
    for _ in range(args.reps):
        fake_grads(params, rng, device, dtype)
        t0 = time.perf_counter()
        opt.step()
        ttnn.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
    host_ms = 1e3 * float(np.median(ts))
    print(f"host step (fp32 masters on host)   median {host_ms:8.2f} ms over {args.reps} reps")

    # --- moreh_adamw on device, same maths, timed the same way
    #     moreh needs every buffer pre-allocated and on device, so the fp32 master, m and v
    #     become DRAM tensors. That is the arrangement adamw_full_precision.cpp uses.
    masters, ms, vs = {}, {}, {}
    for name, t in params.items():
        arr = ft.to_host(t.value).astype(np.float32)
        masters[name] = ft.to_device(arr, device, dtype=ttnn.float32)
        ms[name] = ft.to_device(np.zeros_like(arr), device, dtype=ttnn.float32)
        vs[name] = ft.to_device(np.zeros_like(arr), device, dtype=ttnn.float32)
    step_i = 1

    def moreh_step(step_i):
        for name, t in params.items():
            ttnn.moreh_adamw(masters[name], t.grad, ms[name], vs[name],
                             lr=args.lr, beta1=0.9, beta2=0.999, eps=1e-8,
                             weight_decay=0.01, step=step_i, amsgrad=False,
                             param_out=masters[name], exp_avg_out=ms[name],
                             exp_avg_sq_out=vs[name])
            t.value = ttnn.typecast(masters[name], dtype)

    try:
        fake_grads(params, rng, device, dtype)
        moreh_step(step_i)
        ttnn.synchronize_device(device)
    except Exception as exc:
        print(f"moreh_adamw REFUSED this arrangement: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:160]}")
        print("DECISION: host step. moreh cannot be called on the fp32-master arrangement "
              "at this pin.")
        return host_ms, None
    ts = []
    for i in range(args.reps):
        fake_grads(params, rng, device, dtype)
        t0 = time.perf_counter()
        moreh_step(step_i + 1 + i)
        ttnn.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
    moreh_ms = 1e3 * float(np.median(ts))
    print(f"moreh_adamw step (masters in DRAM) median {moreh_ms:8.2f} ms over {args.reps} reps")

    # --- do they agree? One step from the SAME state, compared in float64.
    rng2 = np.random.default_rng(99)
    p2, _ = adapter_params(device, cfg, dtype, rng2)
    g0 = {}
    for name, t in p2.items():
        shp = tuple(int(d) for d in t.value.shape)
        g0[name] = (rng2.standard_normal(shp) * 1.0 / math.sqrt(shp[0])).astype(np.float32)
        t.grad = ft.to_device(g0[name], device, dtype=dtype)
    init = {name: ft.to_host(t.value).astype(np.float32) for name, t in p2.items()}
    opt2 = ft.AdamW(p2, lr=args.lr)
    opt2.step()
    host_after = {name: opt2.master[name].copy() for name in p2}
    worst = 0.0
    for name in p2:
        m_dev = ft.to_device(init[name], device, dtype=ttnn.float32)
        e_dev = ft.to_device(np.zeros_like(init[name]), device, dtype=ttnn.float32)
        s_dev = ft.to_device(np.zeros_like(init[name]), device, dtype=ttnn.float32)
        gd = ft.to_device(g0[name], device, dtype=dtype)
        ttnn.moreh_adamw(m_dev, gd, e_dev, s_dev, lr=args.lr, beta1=0.9, beta2=0.999,
                         eps=1e-8, weight_decay=0.01, step=1, amsgrad=False,
                         param_out=m_dev, exp_avg_out=e_dev, exp_avg_sq_out=s_dev)
        got = ft.to_host(m_dev).astype(np.float64).reshape(init[name].shape)
        ref = host_after[name].astype(np.float64)
        d = np.linalg.norm(got - ref) / max(np.linalg.norm(ref - init[name]), 1e-30)
        worst = max(worst, float(d))
    print(f"agreement after one step, relative to the UPDATE'S own size: worst {worst:.3e}")
    print(f"DECISION: {'host' if host_ms <= moreh_ms else 'moreh'} step "
          f"({host_ms:.2f} ms vs {moreh_ms:.2f} ms)")
    return host_ms, moreh_ms


def arm_magnitude(device, dtype, args):
    """The control the brief calls non-negotiable: does the step survive the cast?"""
    from tt_bio import train as ft
    print("# --magnitude: the update-magnitude control.")
    print("# `kept` is ||w_after - w_before|| on the DEVICE tensor the forward reads,")
    print("# over ||master_after - master_before||. hallgrad-build measured 0.209 for an")
    print("# eps=0.01 step on logits of order 1; anything well under 1.0 here means the")
    print("# optimiser is an expensive no-op no matter how healthy the gradient looks.")
    cfg = ft.LoraConfig(rank=args.rank)
    print(f"# adapter device dtype: {dtype}. A LoRA A factor is initialised on")
    print(f"# +/- 1/sqrt(in_features), so its elements sit near 6.2e-02 at in=256, where")
    print(f"# bf16 spacing is 2.4e-04 -- the same order as Adam's per-element step at")
    print(f"# lr=3e-4. That proximity is the whole reason this control exists.")
    print(f"{'lr':>9} {'step':>5} {'master step':>13} {'device step':>13} {'kept':>8} "
          f"{'grad norm':>11}  verdict")
    failures = []
    for lr in [float(x) for x in args.lrs.split(",")]:
        rng = np.random.default_rng(17)
        params, _ = adapter_params(device, cfg, dtype, rng)
        opt = ft.AdamW(params, lr=lr)
        for s in range(1, args.steps + 1):
            fake_grads(params, rng, device, dtype)
            rep = opt.step()
            if s not in (1, args.steps):
                continue
            # Aggregate over the whole adapter rather than per tensor: the optimiser takes
            # one step and the question is whether that step lands.
            mn = math.sqrt(sum(r["master_step"] ** 2 for r in rep.values()))
            dn = math.sqrt(sum(r["device_step"] ** 2 for r in rep.values()))
            gn = math.sqrt(sum(r["grad_norm"] ** 2 for r in rep.values()))
            kept = dn / mn if mn > 0 else float("nan")
            # TWO-SIDED, and the upper bound is not symmetry for its own sake. When the
            # master step is below the device dtype's spacing, the device step is a
            # rounding artefact rather than a shrunken version of the real one: a weight
            # sitting near a bf16 boundary jumps a FULL spacing while its neighbour does
            # not move at all, so `kept` can come out well ABOVE 1. The first version of
            # this control gated only on `kept >= 0.95` and therefore passed on kept =
            # 2.683, which is the worst reading in the table. A ratio far from 1 in either
            # direction means the cast is deciding the step.
            ok = 0.95 <= kept <= 1.05
            why = "OK" if ok else ("BELOW bf16 STEP FLOOR" if kept < 0.95
                                   else "ROUNDING INFLATES THE STEP")
            if s == args.steps and not ok:
                failures.append(f"lr={lr:g}: kept={kept:.3f}, the cast is deciding the "
                                f"step ({why})")
            print(f"{lr:>9.1e} {s:>5} {mn:>13.4e} {dn:>13.4e} {kept:>8.3f} {gn:>11.4e}  "
                  f"{why}")
    return failures


def arm_roundtrip(device, dtype, args):
    """Save an adapter mid-training and prove a reload is exact. Cross-process: --verify."""
    from tt_bio import train as ft
    print("# --roundtrip: save the adapter, then reload it and compare.")
    cfg = ft.LoraConfig(rank=args.rank)
    rng = np.random.default_rng(23)
    params, _ = adapter_params(device, cfg, dtype, rng)
    opt = ft.AdamW(params, lr=args.lr)
    for _ in range(args.steps):
        fake_grads(params, rng, device, dtype)
        opt.step()
    # A fixed probe input, so a reload can be checked on a FORWARD and not only on bytes.
    probe = rng.standard_normal((64, 256)).astype(np.float32)
    w = rng.standard_normal((256, 256)).astype(np.float32) / 16.0
    import tt_bio.autograd as ag
    xt = ag.Tensor(ft.to_device(probe, device, dtype=dtype))
    wt = ag.Tensor(ft.to_device(w, device, dtype=dtype))
    out = ft.lora_linear(xt, wt, params["trimul_0.lora_A"], params["trimul_0.lora_B"],
                         scaling=cfg.scaling)
    probe_out = ft.to_host(out.value).astype(np.float64)
    ft.save_adapter(args.path, opt, meta={"steps": opt.steps, "lr": args.lr,
                                          "rank": args.rank})
    np.savez(args.path + ".probe.npz", probe=probe, w=w, out=probe_out,
             sums={n: float(v.sum()) for n, v in opt.master.items()}.__str__())
    size = os.path.getsize(args.path)
    print(f"saved {args.path}  {size:,} B  {len(opt.master)} tensors  after {opt.steps} steps")
    print(f"probe forward checksum {probe_out.sum():.10e}")
    print("now run the same script with --verify to reload in a FRESH process")
    return []


def arm_verify(device, dtype, args):
    from tt_bio import train as ft
    import tt_bio.autograd as ag
    print("# --verify: a FRESH process reloads the adapter and re-runs the probe forward.")
    cfg = ft.LoraConfig(rank=args.rank)
    rng = np.random.default_rng(23)
    params, _ = adapter_params(device, cfg, dtype, rng)
    opt = ft.AdamW(params, lr=args.lr)
    before = ft.to_host(params["trimul_0.lora_B"].value).astype(np.float64)
    meta = ft.load_adapter(args.path, params, device, opt=opt)
    after = ft.to_host(params["trimul_0.lora_B"].value).astype(np.float64)
    moved = float(np.abs(after - before).max())
    z = np.load(args.path + ".probe.npz", allow_pickle=True)
    probe, w, want = z["probe"], z["w"], z["out"]
    xt = ag.Tensor(ft.to_device(probe, device, dtype=dtype))
    wt = ag.Tensor(ft.to_device(w, device, dtype=dtype))
    out = ft.lora_linear(xt, wt, params["trimul_0.lora_A"], params["trimul_0.lora_B"],
                         scaling=cfg.scaling)
    got = ft.to_host(out.value).astype(np.float64)
    exact = bool((got == want).all())
    print(f"metadata {meta}  optimizer steps restored: {opt.steps}")
    print(f"reload moved lora_B by {moved:.4e} (0 would mean the load did nothing)")
    print(f"probe forward: want {want.sum():.10e}  got {got.sum():.10e}")
    print(f"probe forward BIT-IDENTICAL across processes: {exact} "
          f"(max abs diff {np.abs(got - want).max():.3e})")
    failures = []
    if not exact:
        failures.append(f"checkpoint round-trip: probe forward differs by "
                        f"{np.abs(got - want).max():.3e}")
    if moved == 0.0:
        failures.append("checkpoint round-trip: the reload did not change any weight, so "
                        "it proved nothing")
    if opt.steps != args.steps:
        failures.append(f"checkpoint round-trip: optimizer step count {opt.steps} != "
                        f"{args.steps}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moreh", action="store_true")
    ap.add_argument("--magnitude", action="store_true")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lrs", default="1e-5,1e-4,3e-4,1e-3,1e-2")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--path", default="/home/ttuser/ptxft-art/adapter.safetensors")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.path), exist_ok=True)

    import ttnn
    from tt_bio import tenstorrent as tt
    from perf.clocksample import during
    dtype = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[a.dtype]
    device = tt.get_device()
    failures = []
    with during() as clk:
        if a.moreh:
            arm_moreh(device, dtype, a)
            print()
        if a.magnitude:
            failures += arm_magnitude(device, dtype, a)
            print()
        if a.roundtrip:
            failures += arm_roundtrip(device, dtype, a)
            print()
        if a.verify:
            failures += arm_verify(device, dtype, a)
            print()
    print(clk.line(0))
    print()
    if failures:
        print(f"OPTCHECK FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("OPTCHECK PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
