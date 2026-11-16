"""trimul_t2cm_b8 device parity: T2CM over a bfp8 channel-major x must equal T2CM over the same x typecast to bf16
(lossless) bit for bit, at Protenix-v2's c_z 256 (P = 2 stage) and a P = 1 shape. Also times both (warm, 20 calls).

Run on one Blackhole chip: TT_VISIBLE_DEVICES=<chip> python perf/spd_m_protenix6/t2cm_b8_parity.py [N ...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import trimul_tail as TTL  # noqa: E402
from tt_bio import mm_generic as _mm_generic  # noqa: E402


def run(dev, N, K=256, reps=20):
    torch.manual_seed(N)
    up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = torch.randn(1, K, N, N) * torch.logspace(-2, 1, K).reshape(1, K, 1, 1)
    x8 = up(x, ttnn.bfloat8_b)
    x16 = ttnn.typecast(x8, ttnn.bfloat16)
    ln_w, ln_b = up(torch.randn(1, K)), up(torch.randn(1, K))
    wp = up(torch.randn(K, K) / K ** 0.5)
    g = up(torch.randn(1, N, N, K))
    z0 = torch.randn(1, N, N, K)
    ckc = _mm_generic.ckc_args(ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    out, ms = {}, {}
    for name, xc in (("bf16", x16), ("b8", x8)):
        zs = []
        for r in range(reps + 1):
            z = up(z0)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            res = TTL.fused_tail_ln_cm(xc, ln_w, ln_b, wp, g, z, ckc, grid)
            ttnn.synchronize_device(dev)
            if r:
                zs.append(time.perf_counter() - t)
            assert res is not None, (name, dict(TTL.REJECTS))
            if r == 0:
                out[name] = ttnn.to_torch(res).float()
            ttnn.deallocate(z)
        ms[name] = sorted(zs)[len(zs) // 2] * 1e3
    d = (out["bf16"] - out["b8"]).abs()
    return {"N": N, "K": K, "bit_identical": bool(torch.equal(out["bf16"], out["b8"])), "max_abs_diff": d.max().item(),
            "ms_bf16": round(ms["bf16"], 3), "ms_b8": round(ms["b8"], 3)}


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [512, 256]
    dev = T.get_device()
    with T.levers("trimul_t2,trimul_t2cm,trimul_t2cm_b8"):
        for N in sizes:
            print(json.dumps(run(dev, N)), flush=True)


if __name__ == "__main__":
    main()
