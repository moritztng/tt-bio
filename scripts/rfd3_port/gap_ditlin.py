#!/usr/bin/env python3
"""gap_ditlin: the DiT linears at the page fixture's shape, ttnn's default config against swept 2D mcast configs.

x [1, 704, K] bf16 @ w [K, N] bf16 (HiFi4, fp32 dest, packer L1 acc, as the model calls them) for the four DiT shapes
(q/k/v/g 768->1024, fc 768->1536, 1536->768, o 1024->768). Arms: the shipped call (core_grid=BATCH_INVARIANT_GRID),
then MatmulMultiCoreReuseMultiCastProgramConfig over in0_block_w x subblock on the full compute grid, each with a DRAM
and an L1 output. ms/call is the median of N batches of 20 back-to-back calls between syncs; error is max |.| against
float64 and against the shipped output (any config that regroups K moves bits; the fp32 dest keeps it ~1 bf16 ulp).
    python scripts/rfd3_port/gap_ditlin.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio.tenstorrent import get_device                                   # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 15
L, MP = 685, 704
SHAPES = [(768, 1024), (768, 1536), (1536, 768), (1024, 768)]
g = torch.Generator().manual_seed(0)


def timed(fn, dev):
    fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / 20 * 1e3)
    return statistics.median(ts), min(ts)


def configs(kt, nt, mt, grid):
    for gx, gy in {(grid.x, grid.y), (8, 8)}:
        if gx > grid.x or gy > grid.y:
            continue
        pm, pn = -(-mt // gy), -(-nt // gx)
        for bw in (d for d in (1, 2, 3, 4, 6, 8, 12) if kt % d == 0):
            for sh, sw in ((1, 4), (2, 2), (1, 2), (4, 1), (2, 1), (1, 1), (1, 3), (3, 1)):
                if pm % sh or pn % sw:
                    continue
                yield (gx, gy, bw, sh, sw), ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
                    out_subblock_w=sw, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                    fused_activation=None)


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    grid = dev.compute_with_storage_grid_size()
    rec = {"reps": N_REP, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "grid": [grid.x, grid.y], "shapes": {}}
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    for K, N in SHAPES:
        x = torch.randn(1, L, K, generator=g).bfloat16()
        w = (torch.randn(K, N, generator=g) * K ** -0.5).bfloat16()
        ref = x.double() @ w.double()
        xd = up(torch.nn.functional.pad(x, (0, 0, 0, MP - L)))
        wd = up(w)
        ship_fn = lambda: ttnn.linear(xd, wd, compute_kernel_config=ckc, dtype=ttnn.bfloat16,  # noqa: E731
                                      core_grid=M.BATCH_INVARIANT_GRID)
        ship = ttnn.to_torch(ship_fn())[:, :L].double()
        arms = {"ship": dict(zip(("ms", "ms_min"), timed(ship_fn, dev)),
                             err_f64=(ship - ref).abs().max().item(), vs_ship=0.0)}
        for key, pc in configs(K // 32, N // 32, MP // 32, grid):
            for mem in ("dram", "l1"):
                mc = ttnn.L1_MEMORY_CONFIG if mem == "l1" else ttnn.DRAM_MEMORY_CONFIG
                fn = lambda: ttnn.linear(xd, wd, program_config=pc, compute_kernel_config=ckc,  # noqa: E731
                                         dtype=ttnn.bfloat16, memory_config=mc)
                try:
                    o = ttnn.to_torch(fn())[:, :L].double()
                    ms = timed(fn, dev)
                except Exception as e:                       # illegal L1 / subblock combos are expected
                    arms["%dx%d_bw%d_sb%dx%d_%s" % (key + (mem,))] = {"error": str(e)[:120]}
                    continue
                arms["%dx%d_bw%d_sb%dx%d_%s" % (key + (mem,))] = dict(
                    ms=ms[0], ms_min=ms[1], err_f64=(o - ref).abs().max().item(),
                    vs_ship=(o - ship).abs().max().item())
        best = min((a for a in arms.items() if "ms" in a[1]), key=lambda a: a[1]["ms"])
        rec["shapes"][f"{K}x{N}"] = {"best": best[0], "arms": arms}
        print(f"{K}x{N}: ship {arms['ship']['ms']:.4f} ms, best {best[0]} {best[1]['ms']:.4f} ms", flush=True)
        OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
