#!/usr/bin/env python3
"""y-silu-lowering round 2 -- flip DST_ACCUM_MODE inside the FUSED path.

Round 1 established that `ttnn.silu` on 0.67.4 accepts no `compute_kernel_config`, so a standalone
silu cannot be pushed onto the accurate lowering that way. The matmul can be pushed the other way:
`ttnn.linear(activation="silu", compute_kernel_config=<fp32_dest_acc_en=False>)` compiles the SAME
fused kernel with DST_ACCUM_MODE=0, so `calculate_silu` takes _sfpu_exp_21f_bf16_ +
_sfpu_reciprocal_<1> -- the identical function `ttnn.silu` runs.

Accuracy is isolated separately, with no matmul in the way: the same values fed to `ttnn.silu` as
fp32 and as bf16, both scored against a torch fp32 reference of those same values.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch, ttnn


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps=7, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t) / k * 1e6)
    return round(med(out), 3)


def stats(dev_t, ref_t):
    d = dev_t.float() - ref_t.float()
    return dict(max_abs=float(d.abs().max()),
                rel_rmsd=float(d.pow(2).mean().sqrt() / ref_t.float().pow(2).mean().sqrt()),
                pcc=float(np.corrcoef(dev_t.float().flatten().numpy(),
                                      ref_t.float().flatten().numpy())[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "lowering2.json"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1, CG = ttnn.L1_MEMORY_CONFIG, T.CORE_GRID_MAIN
    res = dict(load_start=load(), grid=list(T.COMPUTE_GRID_MAIN), k=a.k)

    ckc_on = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    ckc_off = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=True)

    torch.manual_seed(0)
    xt = torch.randn(1, 30, 298, 256, dtype=torch.bfloat16)
    wt = (torch.randn(256, 1024) * 0.05).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    # ---------- the whole experiment: same fused kernel, DST_ACCUM_MODE flipped ----------
    mm = {}
    for accname, ckc in (("fp32acc_on", ckc_on), ("fp32acc_off", ckc_off)):
        for actname, act in (("bare", None), ("silu", "silu"), ("gelu", "gelu"), ("relu", "relu")):
            def fn(act=act, ckc=ckc):
                y = ttnn.linear(x, w, activation=act, compute_kernel_config=ckc, memory_config=L1,
                                dtype=ttnn.bfloat16, core_grid=CG)
                ttnn.deallocate(y)
            k = f"{accname}_{actname}"
            mm[k] = timed(dev, fn, a.k)
            print("matmul", k, mm[k], flush=True)
    res["matmul"] = mm
    res["penalty"] = {
        f"{acc}_{act}": round(mm[f"{acc}_{act}"] - mm[f"{acc}_bare"], 3)
        for acc in ("fp32acc_on", "fp32acc_off") for act in ("silu", "gelu", "relu")}
    print("penalties", json.dumps(res["penalty"]), flush=True)

    # ---------- standalone eltwise, the reference the penalty is scored against ----------
    y0 = ttnn.linear(x, w, compute_kernel_config=ckc_on, memory_config=L1, dtype=ttnn.bfloat16,
                     core_grid=CG)
    ttnn.synchronize_device(dev)
    el = {}
    for name, op in (("clone", ttnn.clone), ("silu", ttnn.silu), ("gelu", ttnn.gelu),
                     ("relu", ttnn.relu)):
        def fn(op=op):
            z = op(y0, memory_config=L1)
            ttnn.deallocate(z)
        el[name] = timed(dev, fn, max(4, a.k // 2))
        print("eltwise", name, el[name], flush=True)
    res["eltwise"] = el

    # ---------- accuracy of the two silu lowerings, no matmul in the way ----------
    torch.manual_seed(1)
    yb16 = (torch.randn(1, 1, 320, 1024) * 3.0).to(torch.bfloat16)
    ref = torch.nn.functional.silu(yb16.float())
    acc, tim = {}, {}
    for name, dt_torch, dt_ttnn in (("bf16_in", torch.bfloat16, ttnn.bfloat16),
                                    ("fp32_in", torch.float32, ttnn.float32)):
        t = ttnn.from_torch(yb16.to(dt_torch), dtype=dt_ttnn, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        out = ttnn.silu(t, memory_config=L1)
        acc[name] = stats(ttnn.to_torch(out).to(torch.float32), ref)
        # round the fp32 result to bf16 so both are scored at the same output precision too
        acc[name + "_rounded_bf16"] = stats(ttnn.to_torch(out).to(torch.bfloat16).float(), ref)
        def fn(t=t):
            z = ttnn.silu(t, memory_config=L1)
            ttnn.deallocate(z)
        tim[name] = timed(dev, fn, 4)
        ttnn.deallocate(out)
        print("silu", name, "time", tim[name], "acc", json.dumps(acc[name]), flush=True)
    res["silu_accuracy"] = acc
    res["silu_time_by_input_dtype"] = tim

    # ---------- does the bare matmul itself change with fp32_dest_acc_en? ----------
    ref_mm = (xt.float().reshape(-1, 256) @ wt.float()).reshape(1, 30, 298, 1024)
    bare = {}
    for accname, ckc in (("fp32acc_on", ckc_on), ("fp32acc_off", ckc_off)):
        o = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=L1,
                                      dtype=ttnn.bfloat16, core_grid=CG))[:, :, :298, :]
        bare[accname] = stats(o, ref_mm)
    res["bare_matmul_accuracy"] = bare
    print("bare matmul accuracy", json.dumps(bare), flush=True)

    # ---------- fused silu output accuracy at both DST_ACCUM_MODE settings ----------
    ref_act = torch.nn.functional.silu(ref_mm)
    fa = {}
    outs = {}
    for accname, ckc in (("fp32acc_on", ckc_on), ("fp32acc_off", ckc_off)):
        o = ttnn.to_torch(ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                                      memory_config=L1, dtype=ttnn.bfloat16,
                                      core_grid=CG))[:, :, :298, :]
        outs[accname] = o
        fa[accname] = stats(o, ref_act)
    fa["on_vs_off"] = stats(outs["fp32acc_on"], outs["fp32acc_off"])
    fa["on_equals_off"] = bool(torch.equal(outs["fp32acc_on"], outs["fp32acc_off"]))
    res["fused_silu_accuracy"] = fa
    print("fused silu accuracy", json.dumps(fa), flush=True)

    res["load_end"] = load()
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out, flush=True)


main()
