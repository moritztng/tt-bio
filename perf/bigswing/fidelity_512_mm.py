#!/usr/bin/env python3
"""HiFi4 -> HiFi3 at the shapes a 512 aa protenix-v2 fold actually issues.

The fidelity lever has only ever been measured on one shape (trimul.out_proj at 298 aa,
M=8192 K=256 N=256). This prices it at 512 aa across every class that carries the
Pairformer's arithmetic, INCLUDING SDPA, which is 144.0 TFLOP/fold (15.8% of the model)
and which no fidelity measurement has ever touched.

Method follows W4's amortized-region correction: several issues per
synchronize..synchronize region, median of regions, arms alternating inside one process
on one device and one allocator. Accuracy is against an fp32 reference computed from the
SAME bf16 operands the device sees, so operand rounding is outside the error.
"""
import argparse, json, statistics as st, time
from pathlib import Path

import torch
import ttnn

FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi3": ttnn.MathFidelity.HiFi3,
       "HiFi2": ttnn.MathFidelity.HiFi2}


def ckc(fid):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=FID[fid], math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def amort(fn, dev, issues=4, regions=5, warm=2):
    for _ in range(warm):
        o = fn()
        del o
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(regions):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(issues)]
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / issues)
        del outs
    return st.median(ms)


def to_dev(t, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=ttnn.GetDefaultDevice(), memory_config=mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--acc", action="store_true", help="also score accuracy vs fp32 ref")
    a = ap.parse_args()
    N = a.n

    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    grid = dev.compute_with_storage_grid_size()
    res = {"host": "qb2", "ttnn": getattr(ttnn, "__version__", "0.68.0"), "N": N,
           "grid": [grid.x, grid.y], "classes": []}

    def emit():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    torch.manual_seed(0)

    # ---- the matmul classes, each at the 4D/3D shape production issues at N ------------
    # (name, make_operands, run(fid) -> tensor, flops)
    def mm_class(name, xshape, wshape, backend, out_l1=False):
        def build():
            x = to_dev(torch.randn(*xshape))
            w = to_dev(torch.randn(*wshape))
            return x, w
        flop = 2.0
        for d in xshape[:-1]:
            flop *= d
        flop *= xshape[-1] * wshape[-1]
        return (name, build, backend, flop, out_l1)

    CLASSES = [
        mm_class("trimul.in_proj", (1, N, N, 256), (256, 128), "minimal"),
        mm_class("trimul.out_proj", (1, N, N, 256), (256, 256), "minimal"),
        mm_class("triatt.qkv", (N, N, 256), (256, 768), "minimal"),
        mm_class("triatt.g", (N, N, 256), (256, 256), "minimal"),
        mm_class("triatt.out", (N, N, 256), (256, 256), "minimal"),
        mm_class("transition.up", (1, 16, N, 256), (256, 1024), "minimal"),
        mm_class("transition.down", (1, 16, N, 1024), (1024, 256), "minimal", out_l1=True),
    ]

    for name, build, backend, flop, out_l1 in CLASSES:
        try:
            x, w = build()
            mc = ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG
            row = {"class": name, "backend": backend, "gflop": flop / 1e9, "arms": {}}
            refs = {}
            for fid in ("HiFi4", "HiFi3", "HiFi4", "HiFi2", "HiFi3", "HiFi4"):
                c = ckc(fid)
                if backend == "minimal":
                    def fn(c=c):
                        return ttnn.experimental.minimal_matmul(
                            input_tensor=x, weight_tensor=w,
                            compute_kernel_config=c, dtype=ttnn.bfloat16,
                            memory_config=mc)
                else:
                    def fn(c=c):
                        return ttnn.linear(
                            x, w, compute_kernel_config=c, dtype=ttnn.bfloat16,
                            memory_config=mc,
                            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))
                ms = amort(fn, dev)
                row["arms"].setdefault(fid, []).append(round(ms, 5))
                if a.acc and fid not in refs:
                    o = fn()
                    refs[fid] = ttnn.to_torch(o).float()
                    ttnn.deallocate(o)
            for k in row["arms"]:
                row["arms"][k] = {"ms": min(row["arms"][k]), "all": row["arms"][k]}
            base = row["arms"]["HiFi4"]["ms"]
            for k in row["arms"]:
                row["arms"][k]["tflops"] = round(flop / (row["arms"][k]["ms"] * 1e9), 2)
                row["arms"][k]["vs_hifi4"] = round(base / row["arms"][k]["ms"], 4)
            if a.acc and "HiFi4" in refs:
                xt = ttnn.to_torch(x).float()
                wt = ttnn.to_torch(w).float()
                ref = (xt.reshape(-1, xt.shape[-1]) @ wt).reshape(refs["HiFi4"].shape)
                den = ref.pow(2).mean().sqrt().item()
                for k, v in refs.items():
                    d = (v - ref)
                    row.setdefault("acc", {})[k] = {
                        "max_abs": round(d.abs().max().item(), 8),
                        "rel_rms": round((d.pow(2).mean().sqrt().item() / den), 8)}
                # bf16 rounding floor of the reference itself
                rr = ref.bfloat16().float() - ref
                row["acc"]["bf16_output_rounding_floor"] = {
                    "max_abs": round(rr.abs().max().item(), 8),
                    "rel_rms": round(rr.pow(2).mean().sqrt().item() / den, 8)}
            res["classes"].append(row)
            print(name, row["arms"], flush=True)
            ttnn.deallocate(x); ttnn.deallocate(w)
        except Exception as e:  # a class that will not build must not kill the run
            res["classes"].append({"class": name, "error": repr(e)[:300]})
            print("FAIL", name, repr(e)[:200], flush=True)
        emit()

    # ---- the triangle contraction (batched matmul, both operands activations) ----------
    try:
        C = 32
        ap_ = to_dev(torch.randn(1, C, N, N))
        bp_ = to_dev(torch.randn(1, C, N, N))
        flop = 2.0 * C * N * N * N
        row = {"class": "trimul.tri_matmul", "backend": "matmul",
               "gflop": flop / 1e9, "arms": {}}
        for fid in ("HiFi4", "HiFi3", "HiFi4", "HiFi2", "HiFi3", "HiFi4"):
            c = ckc(fid)
            def fn(c=c):
                return ttnn.matmul(ap_, bp_, compute_kernel_config=c,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   dtype=ttnn.bfloat16)
            row["arms"].setdefault(fid, []).append(round(amort(fn, dev, issues=2), 5))
        for k in row["arms"]:
            row["arms"][k] = {"ms": min(row["arms"][k]), "all": row["arms"][k]}
        base = row["arms"]["HiFi4"]["ms"]
        for k in row["arms"]:
            row["arms"][k]["tflops"] = round(flop / (row["arms"][k]["ms"] * 1e9), 2)
            row["arms"][k]["vs_hifi4"] = round(base / row["arms"][k]["ms"], 4)
        res["classes"].append(row)
        print("tri_matmul", row["arms"], flush=True)
        ttnn.deallocate(ap_); ttnn.deallocate(bp_)
    except Exception as e:
        res["classes"].append({"class": "trimul.tri_matmul", "error": repr(e)[:300]})
        print("FAIL tri_matmul", repr(e)[:200], flush=True)
    emit()

    # ---- SDPA, the largest single arithmetic class in the fold -------------------------
    try:
        H, D = 8, 32
        q = to_dev(torch.randn(N, H, N, D))
        k = to_dev(torch.randn(N, H, N, D))
        v = to_dev(torch.randn(N, H, N, D))
        chunk = 256 if N >= 512 else 64
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False)
        flop = 4.0 * N * H * N * N * D  # qk^T + attn@v
        row = {"class": "triatt.sdpa", "backend": "sdpa", "chunk": chunk,
               "gflop": flop / 1e9, "arms": {}}
        for fid in ("HiFi4", "HiFi3", "HiFi4", "HiFi2", "HiFi3", "HiFi4"):
            c = ckc(fid)
            def fn(c=c):
                return ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, program_config=pc,
                    compute_kernel_config=c)
            row["arms"].setdefault(fid, []).append(round(amort(fn, dev, issues=2), 5))
        for kk in row["arms"]:
            row["arms"][kk] = {"ms": min(row["arms"][kk]), "all": row["arms"][kk]}
        base = row["arms"]["HiFi4"]["ms"]
        for kk in row["arms"]:
            row["arms"][kk]["tflops"] = round(flop / (row["arms"][kk]["ms"] * 1e9), 2)
            row["arms"][kk]["vs_hifi4"] = round(base / row["arms"][kk]["ms"], 4)
        res["classes"].append(row)
        print("sdpa", row["arms"], flush=True)
    except Exception as e:
        res["classes"].append({"class": "triatt.sdpa", "error": repr(e)[:300]})
        print("FAIL sdpa", repr(e)[:200], flush=True)
    emit()

    ttnn.close_device(dev)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
