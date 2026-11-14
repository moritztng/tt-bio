"""Boltz-2's token-DiT linears at 512 tokens across math fidelity. The WH census (b2f, 10-13) reads them math-bound at
HiFi4 (0.40-0.50 of the roof), where the same arms on Blackhole were null (abE 10-11). One JSON line per (shape,
fidelity): ms (median of 5 x 50 back-to-back calls after 3 warm, bf16 operands as the fold runs them) and rel_rms vs float64.

usage: TT_VISIBLE_DEVICES=<chip> timeout 600 python perf/spd_ditsdpa/dit_fid.py
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
gen = torch.Generator().manual_seed(0)
# (K, N, calls per fold) from the census rows: qkv, swish / gates / a_to_b, transition out, o-proj
SHAPES = ((768, 3072, 4800), (768, 1536, 14400), (1536, 768, 4800), (768, 768, 9600))
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
x = torch.randn(1, 512, 1536, generator=gen).to(torch.bfloat16)

for K, N, calls in SHAPES:
    xk = x[..., :K].contiguous()
    w = (torch.randn(K, N, generator=gen) / K ** 0.5).to(torch.bfloat16)
    ref = xk.reshape(512, K).double() @ w.double()
    xt = up(xk)
    arms = [(fid, ttnn.bfloat16, "main") for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi")]
    if "--arms" in sys.argv:  # second sweep: weight format and grid at the shipped fidelity
        arms = [("HiFi4", dt, g) for dt in (ttnn.bfloat16, ttnn.bfloat8_b) for g in ("main", "auto")]
    for fid, wdt, g in arms:
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
        wq = ttnn.to_torch(ttnn.typecast(wt, ttnn.bfloat16)).reshape(K, N).double()
        ref = xk.reshape(512, K).double() @ wq
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                               fp32_dest_acc_en=True, packer_l1_acc=True)
        kw = {"core_grid": T.CORE_GRID_MAIN} if g == "main" else {}
        f = lambda: ttnn.linear(xt, wt, compute_kernel_config=ckc, **kw)
        for _ in range(3):
            ttnn.deallocate(f())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(5):  # 50 back-to-back calls per sample: a per-call sync would time the host, not the op
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.deallocate(f())
            ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / 50)
        o = ttnn.to_torch(f()).reshape(512, N).double()
        assert torch.isfinite(o).all()
        rel = float((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        ms = sorted(ts)[2] * 1e3
        print(json.dumps({"K": K, "N": N, "fid": fid, "w": str(wdt).split(".")[-1], "grid": g, "ms": round(ms, 4), "s_per_fold": round(ms * calls / 1e3, 3),
                          "rel_rms": round(rel, 6)}), flush=True)
