#!/usr/bin/env python3
"""Is the 11.52 ms module doing the whole triangle multiplication?

The clock ladder reads the production `TriangleMultiplication` at 11.52 ms/call at 512 aa and
1350 MHz, against the 28.485 ms this campaign carries as ground truth. Before that difference is
allowed to mean anything, the module has to be shown to compute the right answer and to take the
production route. So: a float64 reference at 128 aa, a float32 reference at 512 aa, the executed
op tape, and the branch census the module keeps itself.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.reblock_permute as RB  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
OUT: dict = {}


def reference(z, sd, dtype):
    """`reference.TriangleMultiplicationOutgoing.forward`, verbatim, at `dtype`."""
    z = z.to(dtype)
    w = {k: v.to(dtype) for k, v in sd.items()}
    x = torch.nn.functional.layer_norm(z, (z.shape[-1],), w["norm_in.weight"],
                                       w["norm_in.bias"], 1e-5)
    x_in = x
    x = (x @ w["p_in.weight"].t()) * torch.sigmoid(x @ w["g_in.weight"].t())
    a, b = torch.chunk(x, 2, dim=-1)
    x = torch.einsum("bikd,bjkd->bijd", a, b)
    x = torch.nn.functional.layer_norm(x, (x.shape[-1],), w["norm_out.weight"],
                                       w["norm_out.bias"], 1e-5)
    return (x @ w["p_out.weight"].t()) * torch.sigmoid(x_in @ w["g_out.weight"].t())


def main():
    dev = get_device()
    ckc = T.trunk_compute_kernel_config(ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True))
    for N, c_z, ref_dtype in ((128, 256, torch.float64), (512, 256, torch.float32)):
        torch.manual_seed(0)
        sd = {
            "norm_in.weight": torch.ones(c_z), "norm_in.bias": torch.zeros(c_z),
            "norm_out.weight": torch.ones(c_z), "norm_out.bias": torch.zeros(c_z),
            "p_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
            "g_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
            "p_out.weight": torch.randn(c_z, c_z) * 0.03,
            "g_out.weight": torch.randn(c_z, c_z) * 0.03,
        }
        zt = torch.randn(1, N, N, c_z)
        tm = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)
        z = ttnn.from_torch(zt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                            memory_config=DRAM)
        T.TRIMUL_MM_TRANSPOSE_STATS.clear()
        got = ttnn.to_torch(tm(z)).float()
        want = reference(zt, sd, ref_dtype).float()
        d = (got - want).abs()
        den = want.abs().max().item()
        pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
        row = {"N": N, "c_z": c_z, "ref_dtype": str(ref_dtype),
               "max_abs": d.max().item(), "mean_abs": d.mean().item(),
               "ref_absmax": den, "rel_max": d.max().item() / den, "pcc": pcc,
               "chunk_size": T._trimul_chunk_size(N, tm._hidden, 1),
               "memcfg": str(T._triangle_mul_memory_config(N).buffer_type),
               "branches": {str(k): v for k, v in T.TRIMUL_MM_TRANSPOSE_STATS.items()}}
        mc = T._triangle_mul_memory_config(N)
        if mc.buffer_type == ttnn.BufferType.DRAM:
            cs = T._trimul_inproj_chunk_cap(N, tm._hidden, 1, row["chunk_size"])
            row["chunk_size"] = cs
            row["n_pairs"] = tm._hidden // cs
            row["group"] = T._trimul_inproj_group(N, cs, 1, row["n_pairs"])
        print(f"N={N} c_z={c_z} ref={ref_dtype} max_abs={row['max_abs']:.4e} "
              f"rel={row['rel_max']:.4e} pcc={pcc:.8f} chunk={row['chunk_size']} "
              f"group={row.get('group')} branches={row['branches']}", flush=True)
        OUT[f"parity_N{N}"] = row
        ttnn.deallocate(z)

    # the executed tape at 512 aa, so the composition of the 11.52 ms is on the record
    N, c_z = 512, 256
    torch.manual_seed(0)
    sd = {
        "norm_in.weight": torch.ones(c_z), "norm_in.bias": torch.zeros(c_z),
        "norm_out.weight": torch.ones(c_z), "norm_out.bias": torch.zeros(c_z),
        "p_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
        "g_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
        "p_out.weight": torch.randn(c_z, c_z) * 0.03,
        "g_out.weight": torch.randn(c_z, c_z) * 0.03,
    }
    tm = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ROWS, ON = [], [False]

    def shp(t):
        try:
            return "x".join(str(d) for d in t.shape)
        except Exception:
            return "?"

    def wrap(mod, name, tagger):
        orig = getattr(mod, name)

        def f(*a, **kw):
            if not ON[0]:
                return orig(*a, **kw)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            ttnn.synchronize_device(dev)
            ROWS.append((tagger(a, kw), (time.perf_counter() - t0) * 1e3))
            return out
        setattr(mod, name, f)

    for nm, tag in (("matmul", "matmul"), ("linear", "linear"), ("permute", "permute"),
                    ("transpose", "transpose"), ("layer_norm", "layer_norm"),
                    ("multiply_", "multiply_"), ("chunk", "chunk"), ("concat", "concat"),
                    ("clone", "clone"), ("reallocate", "reallocate"), ("slice", "slice"),
                    ("add", "add"), ("to_memory_config", "to_memory_config")):
        wrap(ttnn, nm, lambda a, kw, tag=tag: f"{tag}[{shp(a[0]) if a else '?'}]")
    wrap(ttnn.experimental, "minimal_matmul",
         lambda a, kw: f"minimal_matmul[{shp(a[0])}@{shp(a[1])}]")
    wrap(RB, "reblock_permute_gated", lambda a, kw: f"reblock_permute_gated[{shp(a[0])}]")
    wrap(RB, "reblock_permute", lambda a, kw: f"reblock_permute[{shp(a[0])}]")
    for _ in range(3):
        ttnn.deallocate(tm(z))
    ON[0] = True
    r = tm(z)
    ON[0] = False
    ttnn.deallocate(r)
    agg: dict = {}
    for k, ms in ROWS:
        e = agg.setdefault(k, [0, 0.0])
        e[0] += 1
        e[1] += ms
    tot = sum(v[1] for v in agg.values())
    print(f"\n=== tape at {N} aa, {len(ROWS)} ops, sum {tot:.3f} ms ===", flush=True)
    for k, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ms:8.3f} ms  {100*ms/tot:5.1f} %  n={n:3d}  {k}", flush=True)
    OUT["tape_512"] = {"sum_ms": tot, "n_ops": len(ROWS),
                       "rows": {k: {"n": v[0], "ms": v[1]} for k, v in agg.items()}}
    OUT["flags"] = {k: getattr(T, k) for k in dir(T)
                    if k.startswith("_TRIMUL") and isinstance(getattr(T, k), (bool, int, str))}
    p = Path(__file__).resolve().parent / "verify_qb2c0.json"
    p.write_text(json.dumps(OUT, indent=1, default=str))
    print(f"\nwrote {p}", flush=True)
    print("FLAGS:", json.dumps(OUT["flags"], default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
