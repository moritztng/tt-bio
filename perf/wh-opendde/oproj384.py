"""OpenDDE's trimul output projection at c_z=384: shipped path vs minimal_matmul block configs.

    TT_VISIBLE_DEVICES=N python perf/wh-opendde/oproj384.py --out OUT.json [--n 736]

The census (cs1, c730 normal) prices the two [1,736,736,384] @ [384,384] projections at 9.5 ms a call, 960 calls
each: ~16.8 TFLOP/s. `_pair_proj_minimal_matmul` only serves kt == 8 (c_z=256), so c_z=384 takes the DRAM linear.
Arms: `shipped` (`_pair_proj_linear` exactly as the trimul calls it, L1 out where it fits), `dram` (its DRAM leg) and `mm:M,K,N,sh,sw`
(ttnn.experimental.minimal_matmul, DRAM out). Under the normal mode's trunk config. us per call (5 calls per batch,
5 batches, median), rel_rms and max abs vs float64, torch.equal vs shipped, AICLK during the arm.
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
ap.add_argument("--configs", default="8,12,12,2,4 8,12,6,2,3 8,6,12,2,4 16,12,12,2,4 16,12,6,4,2 8,4,12,2,4 4,12,12,1,4 "
                "16,6,6,4,2 8,12,4,4,2 16,4,12,2,4", help="space-separated M,K,N,sh,sw; fp32 dest caps sh*sw at 4")
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
g = torch.Generator().manual_seed(0)
N, C = a.n, 384
xh = torch.randn(1, N, N, C, generator=g)
wh = torch.randn(C, C, generator=g) / C ** 0.5
x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ref = ttnn.to_torch(x).double().reshape(-1, C) @ ttnn.to_torch(w).double()
flop = 2 * N * N * C * C
grid = T._mm_core_coord(*T.COMPUTE_GRID_MAIN)
arms = {"shipped": lambda: T._pair_proj_linear(x, w, CKC, ttnn.bfloat16, l1_out=True),
        "dram": lambda: T._pair_proj_linear(x, w, CKC, ttnn.bfloat16)}
for M, K, Nb, sh, sw in (map(int, c.split(",")) for c in a.configs.split()):
    cfg = ttnn.MinimalMatmulConfig(M_block_size=M, K_block_size=K, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
                                   compute_with_storage_grid_size=grid)
    arms[f"mm:{M},{K},{Nb},{sh},{sw}"] = (lambda cfg=cfg: ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=CKC, dtype=ttnn.bfloat16, config=cfg))
res = {"levers": os.environ["TT_BIO_LEVERS"], "shape": [N * N, C, C], "grid": list(T.COMPUTE_GRID_MAIN), "arms": {}}
base = None
for name, fn in arms.items():
    try:
        o = fn()
        got = ttnn.to_torch(o).double().reshape(-1, C)
        ttnn.deallocate(o)
        row = dict(rel_rms=float((got - ref).norm() / ref.norm()), max_abs=float((got - ref).abs().max()))
        if base is None:
            base = got
        row["equal_shipped"] = bool(torch.equal(got, base))
        row.update(timed(fn))
        row["tflops"] = round(flop / row["us_med"] / 1e6, 2)
    except Exception as e:  # noqa: BLE001
        row = dict(err=str(e).splitlines()[0][:200])
    res["arms"][name] = row
    print(json.dumps({name: row}), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
