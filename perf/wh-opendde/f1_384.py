"""trimul_tail F1 at OpenDDE's c_z = 384: is a (12, 12) block that keeps the safe (8, 8) layout correct?

F1 serves only (8, 8) because `_MM_BLOCK[(12, 12)]` = (8, 12, 1, 2, 1) gave wrong numbers at N = 32/64
and hung at N = 128 (trimul_tail.py): out_block doubles and subblock_h halves there. This probe keeps
(8, 8)'s M block 4 and subblock (4, 1) and only widens K to the whole 12-tile contraction.

Per N (small first, so a hang costs the least): production's tail (two `_trimul_out_proj` + gated
multiply_, then the residual add) against F1 EPI 0, EPI 1 and EPI 2 (residual folded in). Reports
equal-to-production, rel RMS to float64, and warm ms per tail. JSON written after every N.

Run: TT_VISIBLE_DEVICES=<chip> python perf/wh-opendde/f1_384.py --out f1.json [--ns 32,64,128,160,736]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import trimul_tail as TTL  # noqa: E402
from tt_bio.af2 import compute_kernel_config  # noqa: E402

CZ = 384


def dev_t(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


CLK = []


def _sample(stop):
    node = Path(f"/sys/class/tenstorrent/tenstorrent!{os.environ.get('TT_VISIBLE_DEVICES', '0')}/tt_aiclk")
    while not stop.wait(0.2):
        try:
            CLK.append(int(node.read_text().split()[0]))
        except (OSError, ValueError):
            pass


def timed(fn, dev, reps):
    """Warm ms per call; AICLK sampled DURING the timed loop into CLK."""
    fn()
    ttnn.synchronize_device(dev)
    stop = threading.Event()
    th = threading.Thread(target=_sample, args=(stop,), daemon=True)
    th.start()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t) / reps * 1e3
    stop.set()
    th.join()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="32,64,128,160,736")
    ap.add_argument("--block", default="4,12,1,4,1")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    dev = T.get_device()
    TTL.F1_BLOCK_KEYS = {(8, 8), (12, 12)}
    TTL._block_for.cache_clear()
    TTL._epi = lambda: TTL.EPI   # the trimul_tail lever would turn EPI 0 into 2; this probe names each
    TTL.set_block(tuple(int(v) for v in a.block.split(",")))
    res = {"block": a.block, "n": {}}
    with T.levers(T.NORMAL_LEVERS):
        ckc_t = T.trunk_compute_kernel_config(compute_kernel_config())
        ckc = MG.ckc_args(ckc_t)
        grid = tuple(T.COMPUTE_GRID_MAIN)
        for n in [int(v) for v in a.ns.split(",")]:
            g = torch.Generator().manual_seed(n)
            bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).float()  # noqa: E731
            xa, xb, z = bf(1, n, n, CZ), bf(1, n, n, CZ), bf(1, n, n, CZ)
            wa, wb = bf(CZ, CZ, sc=CZ ** -0.5), bf(CZ, CZ, sc=CZ ** -0.5)
            d = torch.float64
            ref = (xa.to(d) @ wa.to(d)) * torch.sigmoid(xb.to(d) @ wb.to(d))
            xa_d, xb_d, wa_d, wb_d = dev_t(xa, dev), dev_t(xb, dev), dev_t(wa, dev), dev_t(wb, dev)

            def prod():
                p = T._trimul_out_proj(xa_d, wa_d, ckc_t)
                q = T._trimul_out_proj(xb_d, wb_d, ckc_t)
                y = ttnn.multiply_(p, q, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(q)
                return y

            r = {}
            y = prod()
            yp = ttnn.to_torch(y).float()
            ttnn.deallocate(y)
            r["prod_rel_rms_f64"] = rel_rms(yp, ref)
            for epi in (0, 1):
                TTL.set_epi(epi)
                y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)
                if y is None:
                    r[f"epi{epi}"] = {"declined": {str(k): v for k, v in TTL.REJECTS.items()}}
                    continue
                yf = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
                r[f"epi{epi}"] = {"equal_prod": bool(torch.equal(yf, yp)), "max_abs_vs_prod": (yf - yp).abs().max().item(),
                                  "rel_rms_f64": rel_rms(yf, ref)}
                print(n, epi, r[f"epi{epi}"], flush=True)
            ms_prod = timed(lambda: ttnn.deallocate(prod()), dev, a.reps)
            z_d = dev_t(z, dev)

            def prod_resid():
                y = prod()
                ttnn.add_(z_d, y)
                ttnn.deallocate(y)

            r["ms_prod_with_add"] = timed(prod_resid, dev, a.reps)
            r["ms_prod"] = ms_prod
            TTL.set_epi(1)
            r["ms_f1_epi1"] = timed(lambda: ttnn.deallocate(TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)), dev, a.reps)
            TTL.set_epi(2)
            r["ms_f1_epi2_resid"] = timed(lambda: TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid, resid=z_d), dev, a.reps)
            TTL.set_epi(0)
            r["ms_f1_epi0"] = timed(lambda: ttnn.deallocate(TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)), dev, a.reps)
            r["aiclk_min_med_max"] = [min(CLK), sorted(CLK)[len(CLK) // 2], max(CLK)] if CLK else None
            CLK.clear()
            for t in (xa_d, xb_d, wa_d, wb_d, z_d):
                ttnn.deallocate(t)
            res["n"][n] = r
            print(n, r, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
