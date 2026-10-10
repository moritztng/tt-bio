"""The pair-bias projection ([1,N,N,c_z] @ [c_z,heads]) at each K block `_narrow_proj_linear` can take.

    TT_VISIBLE_DEVICES=N python perf/wh-opendde/narrow384.py --out OUT.json [--n 736]

cs1 (OpenDDE c730 normal) prices [1,736,736,384] @ [384,16] at 7.9 ms a call, 480 calls: ~53 GB/s on a 416 MB read.
`_NARROW_PROJ_BW = 1` holds the K block at one tile, the bit-exact choice. Arms: the cap at 1 (shipped), 2, 4, 8 and
16 (the whole contraction), at c_z 384 and 256 and the heads each model projects. us per call (5 x 5, median),
rel_rms and max abs vs float64, torch.equal vs the shipped arm, AICLK during the arm.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--cases", default="384,16 256,16 384,8", help="space-separated c_z,heads")
ap.add_argument("--caps", default="1 2 4 8 16")
ap.add_argument("--mm", default="", help="space-separated minimal_matmul M,K,N,sh,sw arms (N block 1: one output tile)")
ap.add_argument("--plain", action="store_true", help="also time ttnn.linear with no program config")
a = ap.parse_args()
os.environ.setdefault("TT_BIO_LEVERS", "normal")

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})


def aiclk():
    out = []
    for n in OPENED:
        try:
            out.append(int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return out


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts, clk = [], []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(5):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / 5 * 1e6)
        clk += aiclk()
    return dict(us_med=round(st.median(ts), 1), us_min=round(min(ts), 1), aiclk_min=min(clk) if clk else None,
                aiclk_med=sorted(clk)[len(clk) // 2] if clk else None)


CKC_CLS = ttnn.WormholeComputeKernelConfig if T.is_wormhole() else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers(os.environ["TT_BIO_LEVERS"])
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
N = a.n
res = {"levers": os.environ["TT_BIO_LEVERS"], "n": N, "grid": list(T.COMPUTE_GRID_MAIN), "cases": {}}
for case in a.cases.split():
    C, H = map(int, case.split(","))
    g = torch.Generator().manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, N, N, C, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.randn(C, H, generator=g) / C ** 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev)
    ref = ttnn.to_torch(x).double().reshape(-1, C) @ ttnn.to_torch(w).double()
    rows, base = {}, None
    arms = {}
    for cap in map(int, a.caps.split()):
        arms[f"bw{cap}"] = (lambda cap=cap: (setattr(T, "_NARROW_PROJ_BW", cap),
                                             T._narrow_proj_linear(x, w, CKC, ttnn.bfloat16))[1])
    if a.plain:
        arms["plain"] = lambda: ttnn.linear(x, w, compute_kernel_config=CKC, dtype=ttnn.bfloat16)
    grid = T._mm_core_coord(*T.COMPUTE_GRID_MAIN)
    for M, K, Nb, sh, sw in (map(int, c.split(",")) for c in a.mm.split()):
        cfg = ttnn.MinimalMatmulConfig(M_block_size=M, K_block_size=K, N_block_size=Nb, subblock_h=sh,
                                       subblock_w=sw, compute_with_storage_grid_size=grid)
        arms[f"mm:{M},{K},{Nb},{sh},{sw}"] = (lambda cfg=cfg: ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=CKC, dtype=ttnn.bfloat16, config=cfg))
    for name, fn in arms.items():
        try:
            o = fn()
            got = ttnn.to_torch(o).double().reshape(-1, H)
            ttnn.deallocate(o)
            row = dict(rel_rms=float((got - ref).norm() / ref.norm()), max_abs=float((got - ref).abs().max()))
            base = got if base is None else base
            row["equal_shipped"] = bool(torch.equal(got, base))
            row.update(timed(fn))
            row["gbs"] = round(N * N * C * 2 / row["us_med"] / 1e3, 1)
        except Exception as e:  # noqa: BLE001
            row = dict(err=str(e).splitlines()[0][:200])
        rows[name] = row
        print(json.dumps({case: {name: row}}), flush=True)
    res["cases"][case] = rows
    ttnn.deallocate(x)
    ttnn.deallocate(w)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
