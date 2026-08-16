#!/usr/bin/env python3
"""The Wormhole roofs the ESMFold2 decomposition needs, measured one arm at a time.

`roofs.py` sweeps 15 matmul arms (3 fidelities x 2 accumulators x 2 grids, plus 3 bf8) and writes
its JSON only after all of them return. On the Galaxy one of those arms did not return: the process
sat at 100 % host CPU with nothing on stdout for 2 h 20 m holding the shared benchlock, and because
the write comes last, every arm that HAD completed was lost with it.

So: measure only the roofs the decomposition actually cites, checkpoint the JSON after every single
arm, and drop the full-grid arms, which are the ones that never came back. Run this under a shell
`timeout` -- a hung device op is blocked in C and will not take a Python signal, so the only
reliable bound is the process being killed from outside.

The arms kept are the ones a component can be placed against: the DRAM copy and add roofs, and the
matmul roof at HiFi4 + fp32_dest_acc_en on CORE_GRID_MAIN in both bf16 and bfloat8_b, because
--fast executes bf8 and that is the roof this model runs against.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def bench(fn, n=5, warm=2):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return st.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--mm", type=int, default=4096)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    T.set_fast_mode(a.fast)

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    cls = (ttnn.types.WormholeComputeKernelConfig if T.is_wormhole()
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = lambda fid="HiFi4": cls(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                  math_approx_mode=False, fp32_dest_acc_en=True,
                                  packer_l1_acc=True)
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cores": g.x * g.y, "core_grid_main": list(T.COMPUTE_GRID_MAIN),
         "L": a.L, "mm": a.mm, "fast": a.fast, "wormhole": bool(T.is_wormhole()), "roofs": {}}

    def save(k, v):
        R["roofs"][k] = v
        a.out.write_text(json.dumps(R, indent=1))
        print("%-34s %s" % (k, v), flush=True)

    L, CZ = a.L, 256
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = f(torch.randn(1, L, L, CZ)); zb = f(torch.randn(1, L, L, CZ))
    zbytes = L * L * CZ * 2

    ms = bench(lambda: ttnn.clone(z, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    save("clone_ms", round(ms, 4)); save("clone_GBs", round(2 * zbytes / (ms * 1e-3) / 1e9, 1))
    ms = bench(lambda: ttnn.add(z, zb, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    save("add_ms", round(ms, 4)); save("add_GBs", round(3 * zbytes / (ms * 1e-3) / 1e9, 1))
    ttnn.deallocate(z); ttnn.deallocate(zb)

    N = a.mm
    flop = 2 * N ** 3
    A = f(torch.randn(1, N, N)); B = f(torch.randn(1, N, N))
    for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8")):
        Ax = A if dt is ttnn.bfloat16 else ttnn.typecast(A, dt)
        Bx = B if dt is ttnn.bfloat16 else ttnn.typecast(B, dt)
        for fid in ("HiFi4", "HiFi2", "LoFi"):
            k = "mm%d_%s_%s_fp32acc_main" % (N, name, fid)
            try:
                ms = bench(lambda: ttnn.matmul(Ax, Bx, compute_kernel_config=ckc(fid),
                                               core_grid=T.CORE_GRID_MAIN, dtype=dt))
                save(k + "_ms", round(ms, 4))
                save(k + "_TFLOPs", round(flop / (ms * 1e-3) / 1e12, 2))
            except Exception as e:                                              # noqa: BLE001
                save(k, "ERR %s: %s" % (type(e).__name__, str(e)[:200]))
        if dt is not ttnn.bfloat16:
            ttnn.deallocate(Ax); ttnn.deallocate(Bx)
    ttnn.deallocate(A); ttnn.deallocate(B)

    # Machine balance: the ratio that decides whether naming a component "compute bound" is even
    # possible at its arithmetic intensity.
    tf = R["roofs"].get("mm%d_bf8_HiFi4_fp32acc_main_TFLOPs" % N)
    gb = R["roofs"].get("clone_GBs")
    if isinstance(tf, (int, float)) and isinstance(gb, (int, float)):
        save("balance_FLOP_per_byte", round(tf * 1e12 / (gb * 1e9), 1))
    print("ROOF_WH_DONE", flush=True)


if __name__ == "__main__":
    main()
