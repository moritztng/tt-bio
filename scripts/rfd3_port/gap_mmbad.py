#!/usr/bin/env python3
"""gap_mmbad: where a DiT 768->1536 linear goes wrong against float64, and whether it repeats.

gap_ditlin on WH chip 18 read max |err| 4.0 for the shipped call and every 2D config with in0_block_w in {2,3,4,6}
at this shape, against 0.016 for bw in {1,8,12} and for every other shape and on BH. This reproduces ditlin's exact
inputs (same generator sequence) and a fresh pair, runs the shipped call and two 2D configs three times each, and
records the count, rows, tile coordinates and values of every element off by more than 0.1, plus whether repeated
calls agree bit for bit. The expected cause is the Wormhole fp32-dest cancellation fault (tt_bio/dest_guard.py: any
K block > 1, deterministic, every chip, ~1e-7 of outputs); bw 8 and 12 only dodge it for this input.
    python scripts/rfd3_port/gap_mmbad.py OUT.json
"""
import json, os, pathlib, sys

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio.tenstorrent import get_device                                   # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
L, MP = 685, 704


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    grid = dev.compute_with_storage_grid_size()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    g = torch.Generator().manual_seed(0)
    torch.randn(1, L, 768, generator=g), torch.randn(768, 1024, generator=g)   # ditlin's first shape
    inputs = {"ditlin": (torch.randn(1, L, 768, generator=g).bfloat16(),
                         (torch.randn(768, 1536, generator=g) * 768 ** -0.5).bfloat16())}
    g2 = torch.Generator().manual_seed(1234)
    inputs["fresh"] = (torch.randn(1, L, 768, generator=g2).bfloat16(),
                       (torch.randn(768, 1536, generator=g2) * 768 ** -0.5).bfloat16())

    def pc(bw):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=bw, out_subblock_h=1, out_subblock_w=3,
            per_core_M=-(-MP // 32 // grid.y), per_core_N=-(-48 // grid.x), transpose_mcast=False,
            fused_activation=None)

    rec = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
           "grid": [grid.x, grid.y], "runs": {}}
    for name, (x, w) in inputs.items():
        ref = x.double() @ w.double()
        xd, wd = up(torch.nn.functional.pad(x, (0, 0, 0, MP - L))), up(w)
        arms = {"ship": lambda: ttnn.linear(xd, wd, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                            core_grid=M.BATCH_INVARIANT_GRID),
                "bw4": lambda: ttnn.linear(xd, wd, program_config=pc(4), compute_kernel_config=ckc,
                                           dtype=ttnn.bfloat16),
                "bw8": lambda: ttnn.linear(xd, wd, program_config=pc(8), compute_kernel_config=ckc,
                                           dtype=ttnn.bfloat16)}
        for arm, fn in arms.items():
            outs = [ttnn.to_torch(fn())[0, :L].double() for _ in range(3)]
            err = (outs[0] - ref[0]).abs()
            bad = (err > 0.1).nonzero().tolist()
            r = {"max_err": err.max().item(), "n_bad": len(bad),
                 "repeat_equal": all(torch.equal(outs[0], o) for o in outs[1:]),
                 "repeat_maxdiff": max((outs[0] - o).abs().max().item() for o in outs[1:]),
                 "bad_tiles": sorted({(i // 32, j // 32) for i, j in bad}),
                 "bad_rows": sorted({i for i, _ in bad})[:50], "bad_cols": sorted({j for _, j in bad})[:50],
                 "bad_sample": [(i, j, outs[0][i, j].item(), ref[0, i, j].item()) for i, j in bad[:10]]}
            rec["runs"][f"{name}/{arm}"] = r
            print(f"{name}/{arm}: max {r['max_err']:.4f} n_bad {r['n_bad']} tiles {r['bad_tiles'][:8]} "
                  f"repeat_equal {r['repeat_equal']}", flush=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
