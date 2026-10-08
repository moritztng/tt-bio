"""Which stage of the trimul tail kernel sets its time: a stage ablation at one shape.

`trimul_tail.fused_tail` runs two `[N*N, 256] @ [256, 256]` passes plus the gate. A single bare
pass costs ~3.4 ms at N = 736 whatever the fidelity (HiFi4 and LoFi within 1 %, tail_gemm.py
--fid), so it is not math-bound, and at 161 GB/s it is not at the DRAM roof either. This times the
tail with `TRIMUL_TAIL_ABL` dropping one stage's work at a time while every CB handshake and the
in0 chain stay: 1 no in0 DRAM read, 2 no matmul, 4 no output write, 8 no in1 DRAM read. The stage
whose removal moves the time is the one that binds. The outputs under any bit are garbage.

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/tail_abl.py [--n 736] [--calls 20] [--reps 3]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--abl", default="0,1,2,4,8,9,3,5,6,7,15")
ap.add_argument("--epi", default="0,1")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
g = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
xa = up(torch.randn(1, A.n * A.n, 256, generator=g))
xb = up(torch.randn(1, A.n * A.n, 256, generator=g))
wa = up(torch.randn(256, 256, generator=g) / 16)
wb = up(torch.randn(256, 256, generator=g) / 16)
P = A.n * A.n * 256 * 2

for epi in (int(e) for e in A.epi.split(",")):
    TT.set_epi(epi)
    for abl in (int(a) for a in A.abl.split(",")):
        for shared in (False, True):
            TT.ABL = abl
            b = xa if shared else xb
            rec = {"epi": epi, "abl": abl, "shared": shared}
            try:
                f = lambda: TT.fused_tail(xa, b, wa, wb, ckc, grid)
                y = f(); ttnn.synchronize_device(dev); ttnn.deallocate(y)
                ms = []
                for _ in range(A.reps):
                    t0 = time.perf_counter()
                    outs = [f() for _ in range(A.calls)]
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
                    for o in outs:
                        ttnn.deallocate(o)
                rec["ms"] = round(min(ms), 4)
                rec["spread_ms"] = round(max(ms) - min(ms), 4)
                rec["GBps"] = round((2 if shared else 3) * P / rec["ms"] / 1e6, 1)
            except Exception as e:                                            # noqa: BLE001
                rec["err"] = str(e).splitlines()[0][:200]
            print(json.dumps(rec), flush=True)
TT.ABL = 0
