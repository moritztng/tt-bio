"""pair_tail_cm on one RFD4 trimul call with the float32 stream: served or the reason, and bits against pair_tail.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/tri4/cmdiag.py --weights <bundle> [--n 512]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

BASE = ("trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,"
        "pair_bias_rms,pair_write_res")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--maxp", type=int, default=0, help="T2CM_MAX_P for this arch (bench only)")
    a = ap.parse_args()
    import ttnn
    from safetensors import safe_open
    from tt_bio import tenstorrent as T
    from tt_bio import trimul_tail as TT
    from tt_bio.rfd4.trunk import _trimul_weights, compute_kernel_config

    f = safe_open(str(Path(a.weights) / "model.safetensors"), "pt")
    pre = "nn.pair_update_layers.0."
    state = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
    dev = T.get_device()
    if a.maxp:
        TT.T2CM_MAX_P[T.arch_name()] = a.maxp
    elig = TT.t2cm_eligible

    def traced(*args, **kw):
        why = elig(*args, **kw)
        print("t2cm_eligible ->", why, "copy" if kw.get("copy") is not None else "", flush=True)
        return why
    TT.t2cm_eligible = traced
    n = a.n
    z = torch.randn(1, n, n, 384, generator=torch.Generator().manual_seed(0))
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    with clocksample.during() as clk:
        for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
            outs = {}
            for arm in ("", ",pair_tail_cm"):
                with T.lever_scope(BASE + arm):
                    tri = T.TriangleMultiplication(ending, _trimul_weights(T.WeightScope(state).child(name)),
                                                   compute_kernel_config())
                    s0, c0, r0 = list(TT.T2_STATS), list(TT.T2CM_STATS), dict(TT.REJECTS)
                    z32, zb = up(z, ttnn.float32), up(z, ttnn.bfloat16)
                    out = tri(zb, None, stream32=z32)
                    outs[arm] = (ttnn.to_torch(z32), ttnn.to_torch(zb))
                    print(name, arm or "base", "took_residual", tri.took_residual, "out is stream", out is z32,
                          "T2", [x - y for x, y in zip(TT.T2_STATS, s0)], "T2CM", [x - y for x, y in zip(TT.T2CM_STATS, c0)],
                          "new rejects", {k: v - r0.get(k, 0) for k, v in TT.REJECTS.items() if v != r0.get(k, 0)},
                          flush=True)
                    ts = []
                    for _ in range(a.reps):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        tri(zb, None, stream32=z32)
                        ttnn.synchronize_device(dev)
                        ts.append(time.perf_counter() - t0)
                    print(name, arm or "base", "ms median", round(1e3 * sorted(ts)[len(ts) // 2], 3), flush=True)
                    ttnn.deallocate(z32)
                    ttnn.deallocate(zb)
            a32, ab = outs[""]
            b32, bb = outs[",pair_tail_cm"]
            print(name, "stream max|diff|", float((a32 - b32).abs().max()), "copy max|diff|", float((ab - bb).abs().max()),
                  "update rel", float((a32 - z).norm() / z.norm()), flush=True)
    print(clk.line())


if __name__ == "__main__":
    main()
