#!/usr/bin/env python3
"""y-silu-lowering H1/H2 -- does the fused SILU penalty buy accuracy?

Source (ttnn 0.67.4 wheel, blackhole): `silu_tile` -> llk_math_eltwise_unary_sfpu_silu<APPROX,
DST_ACCUM_MODE> -> calculate_silu<is_fp32_dest_acc_en> -> _sfpu_sigmoid_<is_fp32_dest_acc_en>,
which picks _sfpu_exp_fp32_accurate_ + _sfpu_reciprocal_<2> at DST_ACCUM_MODE=1 and
_sfpu_exp_21f_bf16_ + _sfpu_reciprocal_<1> at 0. Production matmul sets fp32_dest_acc_en=True.
So arm A and y-silu arm B were never the same function. This measures that.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--y-silu-lowering \
        python3 perf/y_silu_lowering/lowering.py --out perf/y_silu_lowering/lowering.json
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
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "lowering.json"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1, CG = ttnn.L1_MEMORY_CONFIG, T.CORE_GRID_MAIN
    gx, gy = T.COMPUTE_GRID_MAIN
    res = dict(load_start=load(), grid=[gx, gy], k=a.k)

    # production compute kernel config, tt_bio/protenix.py:1609
    ckc_prod = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    # same, fp32 dest accumulate OFF -- what an eltwise op with no config is expected to take
    ckc_off = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False)

    torch.manual_seed(0)
    xt = torch.randn(1, 30, 298, 256, dtype=torch.bfloat16)
    wt = (torch.randn(256, 1024) * 0.05).to(torch.bfloat16)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    # ---------- does ttnn.silu take a compute_kernel_config on 0.67.4? ----------
    try:
        probe = ttnn.silu(x, memory_config=L1, compute_kernel_config=ckc_prod)
        ttnn.deallocate(probe)
        res["silu_takes_ckc"] = True
    except Exception as e:
        res["silu_takes_ckc"] = False
        res["silu_ckc_error"] = str(e)[:300]
    print("silu takes compute_kernel_config:", res["silu_takes_ckc"], flush=True)

    # ---------- matmul arms ----------
    mm = {}
    for name, act in (("A_fused_silu", "silu"), ("D_bare", None), ("fused_relu", "relu"),
                      ("fused_gelu", "gelu")):
        def fn(act=act):
            y = ttnn.linear(x, w, activation=act, compute_kernel_config=ckc_prod, memory_config=L1,
                            dtype=ttnn.bfloat16, core_grid=CG)
            ttnn.deallocate(y)
        mm[name] = timed(dev, fn, a.k)
        print("matmul", name, mm[name], flush=True)
    res["matmul"] = mm

    # ---------- standalone eltwise on the fc1 OUTPUT shape ----------
    y0 = ttnn.linear(x, w, compute_kernel_config=ckc_prod, memory_config=L1, dtype=ttnn.bfloat16,
                     core_grid=CG)
    ttnn.synchronize_device(dev)
    res["out_shape"] = list(y0.shape)
    el = {}
    variants = [("clone", ttnn.clone, None), ("silu_default", ttnn.silu, None),
                ("relu_default", ttnn.relu, None), ("gelu_default", ttnn.gelu, None)]
    if res["silu_takes_ckc"]:
        variants += [("silu_fp32acc", ttnn.silu, ckc_prod), ("silu_no_fp32acc", ttnn.silu, ckc_off),
                     ("gelu_fp32acc", ttnn.gelu, ckc_prod), ("relu_fp32acc", ttnn.relu, ckc_prod)]
    for name, op, ckc in variants:
        def fn(op=op, ckc=ckc):
            if ckc is None:
                z = op(y0, memory_config=L1)
            else:
                z = op(y0, memory_config=L1, compute_kernel_config=ckc)
            ttnn.deallocate(z)
        el[name] = timed(dev, fn, max(4, a.k // 2))
        print("eltwise", name, el[name], flush=True)
    res["eltwise"] = el

    # ---------- ACCURACY: same bf16 input, two silu lowerings, one fp32 reference ----------
    torch.manual_seed(1)
    yt = (torch.randn(1, 1, 320, 1024) * 3.0).to(torch.bfloat16)
    yb = ttnn.from_torch(yt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    ref = torch.nn.functional.silu(yt.float())
    acc = {}
    acc["silu_default"] = stats(ttnn.to_torch(ttnn.silu(yb, memory_config=L1)), ref)
    if res["silu_takes_ckc"]:
        acc["silu_fp32acc"] = stats(ttnn.to_torch(
            ttnn.silu(yb, memory_config=L1, compute_kernel_config=ckc_prod)), ref)
        acc["silu_no_fp32acc"] = stats(ttnn.to_torch(
            ttnn.silu(yb, memory_config=L1, compute_kernel_config=ckc_off)), ref)
    res["accuracy_same_input"] = acc
    print("accuracy", json.dumps(acc), flush=True)

    # ---------- ACCURACY at the fold shape ----------
    ref_mm = torch.nn.functional.silu(
        (xt.float().reshape(-1, 256) @ wt.float()).reshape(1, 30, 298, 1024))
    a_out = ttnn.to_torch(ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc_prod,
                                      memory_config=L1, dtype=ttnn.bfloat16,
                                      core_grid=CG))[:, :, :298, :]
    d_out = ttnn.linear(x, w, compute_kernel_config=ckc_prod, memory_config=L1, dtype=ttnn.bfloat16,
                        core_grid=CG)
    b_def = ttnn.to_torch(ttnn.silu(d_out, memory_config=L1))[:, :, :298, :]
    fold = dict(A_fused=stats(a_out, ref_mm), B_silu_default=stats(b_def, ref_mm))
    if res["silu_takes_ckc"]:
        b_fp32 = ttnn.to_torch(ttnn.silu(d_out, memory_config=L1,
                                         compute_kernel_config=ckc_prod))[:, :, :298, :]
        fold["B_silu_fp32acc"] = stats(b_fp32, ref_mm)
        fold["A_vs_B_fp32acc"] = stats(a_out, b_fp32)
        fold["A_equals_B_fp32acc"] = bool(torch.equal(a_out, b_fp32))
    fold["A_vs_B_default"] = stats(a_out, b_def)
    fold["A_equals_B_default"] = bool(torch.equal(a_out, b_def))
    res["accuracy_fold_shape"] = fold
    print("fold-shape accuracy", json.dumps(fold), flush=True)

    res["load_end"] = load()
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out, flush=True)


main()
