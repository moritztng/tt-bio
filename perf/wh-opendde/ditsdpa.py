"""OpenDDE's token-DiT fused SDPA at each (q_chunk, k_chunk): time and error against float64.

    TT_VISIBLE_DEVICES=N python perf/wh-opendde/ditsdpa.py --out OUT.json [--n 1440]

cs1 (c730 normal) prices q,k,v [5,16,1440,64] (head dim 48 padded to 64) with a [1,16,1440,1440] additive bias at
3.82 ms a call, 4800 calls, 18.3 s: ~11 TFLOP/s. The shipped config is q = k = 256, which does not divide 1440
(45 tiles), so both axes pad to 1536. Arms are (q, k) chunk pairs through `fused_sdpa`, the fold's own entry point.
Inputs mimic the fold: q, k, v with the 16 zero-padded head-dim columns, scale 48**-0.5, bias N(0, 2).
us per call (5 x 5, median), rel_rms and max abs vs a float64 softmax attention, AICLK during the arm.
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
ap.add_argument("--n", type=int, default=1440)
ap.add_argument("--chunks", default="256,256 288,288 160,160 480,480 288,160 160,288 480,288 288,480 96,480 "
                "480,160 128,128 512,512 256,128 128,256")
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


B, H, N, D, DL = 5, 16, a.n, 64, 48
g = torch.Generator().manual_seed(0)
qkv = [torch.zeros(B, H, N, D) for _ in range(3)]
for t in qkv:
    t[..., :DL] = torch.randn(B, H, N, DL, generator=g)
bias = 2 * torch.randn(1, H, N, N, generator=g)
dq, dk, dv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in qkv)
db = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
q64, k64, v64 = (ttnn.to_torch(t).double() for t in (dq, dk, dv))
b64 = ttnn.to_torch(db).double()
scale = DL ** -0.5
# SDPA scales its additive mask along with QK: softmax(scale * (q k^T + bias)).
ref = torch.softmax(scale * (q64 @ k64.transpose(-1, -2) + b64), dim=-1) @ v64
flop = 4 * B * H * N * N * D
res = {"shape": [B, H, N, D], "grid": list(T.COMPUTE_GRID_MAIN), "arms": {}}
for c in a.chunks.split():
    qc, kc = map(int, c.split(","))
    cfg = T._sdpa_program_config(q_chunk_size=qc, k_chunk_size=kc)
    fn = lambda cfg=cfg: T.fused_sdpa(dq, dk, dv, attn_mask=db, scale=scale, program_config=cfg)  # noqa: E731
    try:
        o = fn()
        got = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        row = dict(rel_rms=float((got - ref).norm() / ref.norm()), max_abs=float((got - ref).abs().max()))
        row.update(timed(fn))
        row["tflops"] = round(flop / row["us_med"] / 1e6, 2)
    except Exception as e:  # noqa: BLE001
        row = dict(err=str(e).splitlines()[0][:200])
    res["arms"][f"q{qc},k{kc}"] = row
    print(json.dumps({f"q{qc},k{kc}": row}), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
