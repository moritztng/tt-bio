"""minimal_matmul block sweep for RFD4's pair-track linears, bf16 and bfp8, against ttnn.linear.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/bfp8/mm_sweep.py --weights <bundle> --out <json> \
        [--n 128 512] [--sites tr_fc1,tr_fc2,tm_gout,tm_pout] [--reps 15]

`pair_mm` (tt-bio's pair-track minimal_matmul) has no block for RFD4's shapes, so its calls keep
ttnn's plan. This sweeps (M, K, N block, subblock) per site and dtype arm, times each config as
the median of `--reps` synced calls, and grades the best one per arm against float64 the way
sites.py does (row sample, checkpoint layer 0 weights). The fidelity is HiFi4 on bf16 operands
and HiFi2 where both operands are bfp8 (`bfp8_fidelity`, the same bits).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
from sites import ROWS, inputs, sites  # noqa: E402

ARMS = [  # name, x dtype, w dtype, out dtype, fidelity
    ("bf16", "bf16", "bf16", "bf16", "HiFi4"),
    ("w8", "bf16", "b8", "bf16", "HiFi2"),
    ("w8_o8", "bf16", "b8", "b8", "HiFi2"),
    ("x8w8", "b8", "b8", "bf16", "HiFi2"),
    ("x8w8_o8", "b8", "b8", "b8", "HiFi2"),
]


def blocks(kt, nt, rows_t):
    for M, Kb, Nb in itertools.product((4, 8, 16), (2, 3, 4, 6, 8, 12), (1, 2, 3, 4, 6, 8)):
        if kt % Kb or nt % Nb or rows_t % M:
            continue
        for sh, sw in ((4, 1), (2, 2), (1, 4), (2, 1), (1, 2), (8, 1), (1, 8), (4, 2), (2, 4)):
            # fp32 dest: 4 tiles per subblock at most (8 when the kernel runs half-sync is not assumed)
            if M % sh or Nb % sw or sh * sw > 4:
                continue
            yield M, Kb, Nb, sh, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--sites", default="tr_fc1,tr_fc2,tm_gout,tm_pout,tm_in")
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()

    import ttnn

    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, _mm_core_coord, get_device

    dev = get_device()
    base = compute_kernel_config()
    dt = {"bf16": ttnn.bfloat16, "b8": ttnn.bfloat8_b}
    grid = _mm_core_coord(*COMPUTE_GRID_MAIN)

    def ckc(fid):
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=base.math_approx_mode,
            fp32_dest_acc_en=base.fp32_dest_acc_en, packer_l1_acc=base.packer_l1_acc)

    up = lambda t, d: ttnn.from_torch(t.float().contiguous(), dtype=dt[d], layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def timed(fn, reps):
        ts = []
        for _ in range(reps + 2):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            y = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(y)
        ts = sorted(ts[2:])
        return 1e3 * ts[len(ts) // 2]

    S = {s[0]: s for s in sites(a.weights)}
    w_fc1, b_fc1 = S["tr_fc1"][1], S["tr_fc1"][2]
    want = set(a.arms.split(","))
    res = []
    with clocksample.during() as clk:
        for n in a.n:
            for name in a.sites.split(","):
                _, w, b, act = S[name]
                k, m = int(w.shape[1]), int(w.shape[0])
                x = inputs(name, n, k, w_fc1, b_fc1, torch.Generator().manual_seed(0))
                ref = x[:ROWS].double() @ w.t() + (0 if b is None else b)
                if act == "relu":
                    ref = torch.relu(ref)
                for arm, xd, wd, od, fid in ARMS:
                    if arm not in want:
                        continue
                    xt, wt = up(x, xd), up(w.t(), wd)
                    bt = None if b is None else up(b[None], "bf16")
                    c = ckc(fid)

                    def err_of(y):
                        got = ttnn.to_torch(y).reshape(-1, m)[:ROWS].double()
                        ttnn.deallocate(y)
                        return float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())

                    lin = lambda: ttnn.linear(xt, wt, bias=bt, compute_kernel_config=c, dtype=dt[od],  # noqa: E731
                                              activation=act, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    row = dict(n=n, site=name, arm=arm, k=k, m=m)
                    try:
                        row["linear_ms"] = timed(lin, a.reps)
                        row["linear_err"] = err_of(lin())
                    except RuntimeError as e:
                        row["linear_refused"] = str(e).splitlines()[0][:120]
                    best = None
                    for blk in blocks(k // 32, m // 32, n * n // 32):
                        cfg = ttnn.MinimalMatmulConfig(
                            M_block_size=blk[0], K_block_size=blk[1], N_block_size=blk[2],
                            subblock_h=blk[3], subblock_w=blk[4], compute_with_storage_grid_size=grid)
                        mm = lambda: ttnn.experimental.minimal_matmul(  # noqa: E731
                            input_tensor=xt, weight_tensor=wt, bias_tensor=bt, compute_kernel_config=c,
                            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if act else None,
                            dtype=dt[od], config=cfg)
                        try:
                            ms = timed(mm, 4)
                        except RuntimeError:
                            continue
                        if best is None or ms < best[0]:
                            best = (ms, blk, mm)
                    if best is not None:
                        row["mm_block"] = list(best[1])
                        row["mm_ms"] = timed(best[2], a.reps)
                        row["mm_err"] = err_of(best[2]())
                    for t in (xt, wt, bt):
                        if t is not None:
                            ttnn.deallocate(t)
                    row["t"] = time.strftime("%H:%M:%S")
                    res.append(row)
                    print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(dict(rows=res, aiclk=clk.summary(), load=clk.load and max(clk.load),
                                           script="perf/rfd4/bfp8/mm_sweep.py"), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
