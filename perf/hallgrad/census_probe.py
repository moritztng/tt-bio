#!/usr/bin/env python3
"""Same-shape controls for the census's three largest families. Measurement only, nothing changed.

Each row times the shipped call as the census saw it next to the same arithmetic in another
form that ttnn already offers, so the census's roof column can quote an achievable measured
rate beside the physical one. 3 warm-up calls, 20 timed, one synchronize_device, 5 repeats.
"""
import json, pathlib, statistics, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402
from perf.hallgrad.census import stamp  # noqa: E402


def main():
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    dev = tt.get_device()
    clocks = ClockTrace(period=1.0).start()
    cfg = ag.precise_config()
    T = lambda *s: ttnn.from_torch(torch.randn(*s).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev)
    x3, x2, w = T(256, 256, 128), T(65536, 128), T(128, 128)
    g3 = T(256, 256, 128)
    p = T(128, 4, 128, 256)          # probabilities, one chunk
    v = T(128, 4, 256, 32)
    q = T(128, 4, 128, 32)
    dOut = T(128, 4, 128, 32)
    h = T(256, 256, 128)
    pc = T(128, 256, 256)
    cases = {
        "linear_3d_hifi4 [256,256,128]@[128,128]": lambda: ttnn.linear(x3, w, compute_kernel_config=cfg),
        "linear_2d_hifi4 [65536,128]@[128,128]": lambda: ttnn.linear(x2, w, compute_kernel_config=cfg),
        "matmul_3d_hifi4 [256,256,128]@[128,128]": lambda: ttnn.matmul(x3, w, compute_kernel_config=cfg),
        "matmul_2d_hifi4 [65536,128]@[128,128]": lambda: ttnn.matmul(x2, w, compute_kernel_config=cfg),
        "matmul_3d_tb_hifi4 [256,256,128]@[128,128]^T": lambda: ttnn.matmul(g3, w, transpose_b=True, compute_kernel_config=cfg),
        "reshape3d_then_linear_2d": lambda: ttnn.linear(ttnn.reshape(x3, [65536, 128]), w, compute_kernel_config=cfg),
        "triatt_PV [128,4,128,256]@[128,4,256,32]": lambda: ttnn.matmul(p, v, compute_kernel_config=cfg),
        "triatt_PtdO [128,4,128,256]^T@[128,4,128,32]": lambda: ttnn.matmul(p, dOut, transpose_a=True, compute_kernel_config=cfg),
        "triatt_QKt [128,4,128,32]@[128,4,256,32]^T": lambda: ttnn.matmul(q, v, transpose_b=True, compute_kernel_config=cfg),
        "sdpa_fwd_chunk q[128,4,128,32] kv[128,4,256,32]": lambda: ttnn.transformer.scaled_dot_product_attention(q, v, v, is_causal=False),
        "reshape_heads [256,256,128]->[256,256,4,32]": lambda: ttnn.reshape(h, [256, 256, 4, 32]),
        "permute_201 [256,256,128]": lambda: ttnn.permute(h, (2, 0, 1)),
        "permute_120 [128,256,256]": lambda: ttnn.permute(pc, (1, 2, 0)),
        "clone [256,256,128]": lambda: ttnn.clone(h),
    }
    out = {"stamp": stamp()}
    for name, fn in cases.items():
        try:
            ts = []
            for _ in range(5):
                for _ in range(3):
                    fn()
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(20):
                    fn()
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) / 20)
            out[name] = {"median_us": statistics.median(ts) * 1e6, "min_us": min(ts) * 1e6,
                         "max_us": max(ts) * 1e6}
            print(f"{name:52s} median {out[name]['median_us']:8.1f} us  min {out[name]['min_us']:8.1f}  max {out[name]['max_us']:8.1f}", flush=True)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{name:52s} FAILED {out[name]['error']}", flush=True)
    clocks.stop()
    out["clock_window"] = clocks.window(0, time.time() + 1)
    out["clock_samples"] = clocks.samples
    print("AICLK", out["clock_window"])
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
