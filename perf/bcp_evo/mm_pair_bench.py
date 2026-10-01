#!/usr/bin/env python3
"""Pair-track matmuls at [1,S,S,K] x [K,N]: the shipped `ttnn.matmul`/`ttnn.linear` against
`experimental.minimal_matmul`, unconfigured and at a few block configs, graded against float64.

The block census (out/prof_blk) has these on `bmm_large_block` at 130-150 GB/s: pair-transition
fc1/fc2 forward and the 2-D-weight dx in the backward (`transpose_b`). minimal_matmul takes no
transpose, so its dx arm is given the transposed weight up front (weights are tiny and could be
cached). Timed synced, arms alternating every rep.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=288)
    ap.add_argument("--cases", default="128x128,128x512,512x128,128x1024,1024x128,128x384,384x128")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/mm_pair_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    grid = T._mm_core_coord(*T.COMPUTE_GRID_MAIN)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {"S": a.s, "grid": list(T.COMPUTE_GRID_MAIN)}
    for spec in a.cases.split(","):
        K, N = (int(v) for v in spec.split("x"))
        kt, nt = K // 32, N // 32
        torch.manual_seed(0)
        x = torch.randn(1, a.s, a.s, K).bfloat16()
        w = (torch.randn(K, N) / K ** 0.5).bfloat16()
        xd, wd, wtd = up(x), up(w), up(w.t().contiguous())
        ref = x.double() @ w.double()
        fns = {"linear": lambda: ttnn.linear(xd, wd, compute_kernel_config=ckc,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG),
               "matmul_tb": lambda: ttnn.matmul(xd, wtd, transpose_b=True,
                                                compute_kernel_config=ckc),
               "mm_none": lambda: ttnn.experimental.minimal_matmul(
                   input_tensor=xd, weight_tensor=wd, compute_kernel_config=ckc)}
        for M in (4, 8):
            for Kb in sorted({min(kt, 8), min(kt, 4)}):
                if kt % Kb:
                    continue
                for Nb, sh, sw in ((1, 4, 1), (2, 2, 2), (4, 1, 4)):
                    if nt % Nb:
                        continue
                    cfg = ttnn.MinimalMatmulConfig(M_block_size=M, K_block_size=Kb,
                                                   N_block_size=Nb, subblock_h=sh, subblock_w=sw,
                                                   compute_with_storage_grid_size=grid)
                    fns[f"mm_{M}_{Kb}_{Nb}_{sh}{sw}"] = (
                        lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                            input_tensor=xd, weight_tensor=wd, compute_kernel_config=ckc,
                            config=cfg))
        out, base = {}, None
        for arm in list(fns):
            try:
                yt = fns[arm]()
                y = down(yt).reshape(ref.shape)
            except Exception as e:  # a config the kernel refuses is a result, not a crash
                print(f"{spec} {arm} refused: {str(e)[:120]}", flush=True)
                del fns[arm]
                continue
            if base is None:
                base = y
            out[arm] = {"rel_l2": float((y - ref).norm() / ref.norm()),
                        "equal_linear": bool(torch.equal(y, base)), "t": []}
        for rep in range(a.reps):
            for arm in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = fns[arm]()
                ttnn.synchronize_device(dev)
                out[arm]["t"].append(time.perf_counter() - t0)
                del r
        for arm in fns:
            ts = sorted(out[arm].pop("t"))
            out[arm].update(median_ms=round(1e3 * ts[len(ts) // 2], 4), min_ms=round(1e3 * ts[0], 4))
        res[spec] = out
        best = min((v["median_ms"], k) for k, v in out.items() if k.startswith("mm"))
        print(spec, "linear", out["linear"]["median_ms"], "matmul_tb", out["matmul_tb"]["median_ms"],
              "best", best, json.dumps({k: (v["median_ms"], round(v["rel_l2"], 6), v["equal_linear"])
                                        for k, v in out.items()}), flush=True)
    res["window_utc"] = time.time()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
