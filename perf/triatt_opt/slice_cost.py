#!/usr/bin/env python3
"""Does the fused N=1056 projection survive the cost of extracting its columns?

state/triatt-absolute-optimal.md 4.4 measured the MATMUL only (2.598 fused vs 3.456 separate,
1.330x, bit-exact). The consumers need [0:768] for nlp_create_qkv_heads, [768:1024] for the
gate multiply and [1024:1056] for the bias. In ttnn a last-dim slice of a TILE_LAYOUT tensor is
a copy, not a view.

PREDICTION, WRITTEN BEFORE THE RUN: the [0:768] slice alone reads 384 MiB and writes 384 MiB,
so at the measured 388.9 GB/s copy roof it is ~2.07 ms at 512 aa and the fusion is a NET LOSS
of ~1.2 ms/call once extraction is counted. If so, 4.4's 1.387x is a matmul-only number that
does not survive integration, and only the two config levers land.
Second prediction: the L1-destination transpose (_TRANSPOSE_L1_HEADROOM lowered) does not fire
at 512 aa at any headroom, because the pair tensor is 134.22 MB of a 168.6 MB grid.
"""
import json, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device

RES = {"predictions": __doc__}


def timed(fn, warm=2, reps=5):
    dev = T.get_device()
    for _ in range(warm):
        r = fn(); del r
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        del r
    return st.median(ts) * 1e3


def main():
    dev = get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)
    RES["loadavg"] = open("/proc/loadavg").read().strip()
    out = []
    for S in (320, 512):
        mt = S * (-(-S // 32))
        f = ttnn.from_torch(torch.randn(S, S, 1056).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        row = {"S": S}
        row["slice_qkv_ms"] = timed(lambda: ttnn.slice(f, [0, 0, 0], [S, S, 768]))
        row["slice_g_ms"] = timed(lambda: ttnn.slice(f, [0, 0, 768], [S, S, 1024]))
        row["slice_bias_ms"] = timed(lambda: ttnn.slice(f, [0, 0, 1024], [S, S, 1056]))
        row["slice_total_ms"] = row["slice_qkv_ms"] + row["slice_g_ms"] + row["slice_bias_ms"]
        ttnn.deallocate(f)
        out.append(row)
        print("SLICE", json.dumps(row), flush=True)
    RES["slices"] = out

    # --- the transpose, at every headroom that could fire -------------------------------------
    tr = []
    for S in (320, 512):
        z = ttnn.from_torch(torch.randn(S, S, 256).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        row = {"S": S, "MB": S * S * 256 * 2 / 1e6}
        row["shipped_ms"] = timed(lambda: T._pair_transpose(z, T._transpose_memory_config(z)))
        got = {}
        for hr in (2.5, 1.6, 1.25, 1.0, 0.8):
            T._TRANSPOSE_L1_HEADROOM = hr
            mc = T._transpose_memory_config(z)
            is_l1 = mc is not None and "L1" in str(mc)
            if not is_l1:
                got[str(hr)] = "dram"
                continue
            try:
                got[str(hr)] = timed(lambda mc=mc: T._pair_transpose(z, mc))
            except Exception as e:
                got[str(hr)] = f"refused: {type(e).__name__}"
        T._TRANSPOSE_L1_HEADROOM = 2.5
        row["by_headroom"] = got
        ttnn.deallocate(z)
        tr.append(row)
        print("TRANSPOSE", json.dumps(row), flush=True)
    RES["transpose"] = tr
    Path("perf/triatt_opt/slice_cost.json").write_text(json.dumps(RES, indent=1))
    print("wrote perf/triatt_opt/slice_cost.json")


if __name__ == "__main__":
    main()
