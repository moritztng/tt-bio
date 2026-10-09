"""OpenDDE's c=384 pair transition fc1 in fast mode: why bfp8 + fused silu costs 350 us against 130 in bf16.

    TT_VISIBLE_DEVICES=N python perf/wh-opendde/silu_b8.py --out OUT.json [--silu-f32]

One row block [4*736, 384] @ [384, 1536] under the trunk config (HiFi4 base + the mode's fidelity lever), input in L1.
Arms: output dtype bf16 / bfp8 x silu fused in the matmul / unfused (ttnn.silu after) / none. us per call (10 calls
per batch, 7 batches, median), rel_rms vs float64, AICLK during the arm.
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
ap.add_argument("--silu-f32", action="store_true")
ap.add_argument("--rows", type=int, default=4 * 736)
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = "fast" + (",silu_f32" if a.silu_f32 else "")

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
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(10):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / 10 * 1e6)
        clk += aiclk()
    return dict(us_med=round(st.median(ts), 2), us_min=round(min(ts), 2), aiclk_min=min(clk) if clk else None,
                aiclk_med=sorted(clk)[len(clk) // 2] if clk else None)


CKC_CLS = ttnn.WormholeComputeKernelConfig if T.is_wormhole() else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers(os.environ["TT_BIO_LEVERS"])
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
SILU_CKC = T.silu_ckc(CKC)
g = torch.Generator().manual_seed(0)
M, K, N = a.rows, 384, 1536
res = {"levers": os.environ["TT_BIO_LEVERS"], "shape": [M, K, N], "arms": {}}
for wdt_name, wdt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
    xh = torch.randn(1, 1, M, K, generator=g)
    wh = torch.randn(K, N, generator=g) / K ** 0.5
    x = ttnn.from_torch(xh, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    w = ttnn.from_torch(wh, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.to_torch(x).double().view(M, K) @ ttnn.to_torch(w).double().view(K, N)
    for odt_name, odt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        for act in ("fused", "unfused", "none"):
            def fn(act=act, odt=odt):
                kw = dict(memory_config=ttnn.L1_MEMORY_CONFIG, dtype=odt, core_grid=T.CORE_GRID_MAIN)
                if act == "fused":
                    return ttnn.linear(x, w, activation="silu", compute_kernel_config=SILU_CKC, **kw)
                o = ttnn.linear(x, w, compute_kernel_config=CKC, **kw)
                return ttnn.silu(o, output_tensor=o) if act == "unfused" else o
            arm = f"in{wdt_name}:out{odt_name}:{act}"
            try:
                o = fn()
                got = ttnn.to_torch(o).double().view(M, N)
                ttnn.deallocate(o)
                r = ref if act == "none" else torch.nn.functional.silu(ref)
                row = dict(rel_rms=float((got - r).norm() / r.norm()))
                row.update(timed(fn))
            except Exception as e:  # noqa: BLE001
                row = dict(err=str(e).splitlines()[0][:200])
            res["arms"][arm] = row
            print(json.dumps({arm: row}), flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(w)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
