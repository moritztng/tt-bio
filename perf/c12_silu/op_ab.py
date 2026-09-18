"""Op-level paired A/B for the fused-silu epilogue on Blackhole (c12-unfused-silu-bh).

Seven arms interleaved rep by rep in ONE process on ONE device open, so a constant co-tenant load
cancels. The clock is FORCED to a fixed MHz for the whole session and sampled at 500 Hz in a
SUBPROCESS -- a sampler thread gets one sample per timed interval because ttnn holds the GIL across
a device call (see clk.Sampler).

Known-answer controls:
  * FLOPs and bytes counted from logical dims AND from the tile grid; the two must agree exactly,
    which proves the shape is tile-aligned and the counts are right.
  * `fc1_fused` and `fc1_plain` must reproduce c12-kblock-unlock's sweep prices of 0.1201 and
    0.0314 ms/call. If they do not, the program config is not the production one and nothing below
    is comparable.
  * `dispatch_floor` runs a [1,1,32,32] in-place silu, one tile, the same number of times. Whatever
    it reads is host dispatch, not arithmetic, and it bounds how much of every other arm is dispatch.

The A/A arm is `fc1_fused` run a second time later in the same rep sequence. Its ratio is this
session's own noise floor. A ratio smaller than its own session floor is not a result.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clk

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

CALLS = int(os.environ.get("C12_CALLS", "500"))
REPS = int(os.environ.get("C12_REPS", "10"))
MHZ = int(os.environ.get("C12_MHZ", "1350"))
WARM = 2
OUT = Path(sys.argv[1])

# The production fc1 in Transition.swiglu at 512 aa: a [1,16,512,128] x w [128,512] -> [1,16,512,512]
B, H, M, K, N = 1, 16, 512, 128, 512
TILE = 32
CALLS_PER_FOLD = 8960


def counts():
    """FLOPs and bytes two ways. The two must agree exactly."""
    flops_logical = 2 * B * H * M * K * N
    bytes_logical = (B * H * M * K + K * N + B * H * M * N) * 2
    ta = B * H * (M // TILE) * (K // TILE)
    tw = (K // TILE) * (N // TILE)
    to = B * H * (M // TILE) * (N // TILE)
    flops_tiles = 2 * to * TILE * TILE * K
    bytes_tiles = (ta + tw + to) * TILE * TILE * 2
    silu_el = B * H * M * N
    return {
        "fc1_flops_logical": flops_logical, "fc1_flops_tiles": flops_tiles,
        "fc1_bytes_logical": bytes_logical, "fc1_bytes_tiles": bytes_tiles,
        "fc1_flops_agree": flops_logical == flops_tiles,
        "fc1_bytes_agree": bytes_logical == bytes_tiles,
        # in-place silu: one read + one write of the [1,16,512,512] bf16 tensor
        "silu_bytes_logical": 2 * silu_el * 2,
        "silu_bytes_tiles": 2 * to * TILE * TILE * 2,
        "silu_bytes_agree": (2 * silu_el * 2) == (2 * to * TILE * TILE * 2),
        "silu_elements": silu_el,
        "tiles_a": ta, "tiles_w": tw, "tiles_out": to,
    }


def main():
    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    if len(nodes) != 1:
        raise SystemExit(f"expected exactly one open chip, got {nodes}")
    node = nodes[0]
    L1 = ttnn.L1_MEMORY_CONFIG
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    a = ttnn.to_device(ttnn.from_torch(torch.randn((B, H, M, K), dtype=torch.bfloat16),
                                       layout=ttnn.TILE_LAYOUT), dev, memory_config=L1)
    w = ttnn.to_device(ttnn.from_torch(torch.randn((K, N), dtype=torch.bfloat16),
                                       layout=ttnn.TILE_LAYOUT), dev, memory_config=L1)
    x0_t = torch.randn((B, H, M, N), dtype=torch.bfloat16)
    x0 = ttnn.to_device(ttnn.from_torch(x0_t, layout=ttnn.TILE_LAYOUT), dev, memory_config=L1)
    x = ttnn.to_device(ttnn.from_torch(x0_t.clone(), layout=ttnn.TILE_LAYOUT), dev, memory_config=L1)
    tiny = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, TILE, TILE), dtype=torch.bfloat16),
                                          layout=ttnn.TILE_LAYOUT), dev, memory_config=L1)

    def fc1(act):
        o = ttnn.linear(a, w, activation=act, compute_kernel_config=ckc,
                        memory_config=L1, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(o)

    def fc1_unfused():
        o = ttnn.linear(a, w, activation=None, compute_kernel_config=ckc,
                        memory_config=L1, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.silu(o, memory_config=L1, output_tensor=o)
        ttnn.deallocate(o)

    ARMS = {
        "fc1_fused": lambda: fc1("silu"),
        "silu": lambda: ttnn.silu(x, memory_config=L1, output_tensor=x),
        "fc1_plain": lambda: fc1(None),
        "fc1_unfused": fc1_unfused,
        "fc1_fused_aa": lambda: fc1("silu"),
        "dispatch_floor": lambda: ttnn.silu(tiny, memory_config=L1, output_tensor=tiny),
        "hostloop": lambda: None,
    }
    ORDER = ["fc1_fused", "silu", "fc1_plain", "fc1_unfused", "fc1_fused_aa",
             "dispatch_floor", "hostloop"]

    for _ in range(WARM):
        for name in ORDER:
            for _ in range(2):
                ARMS[name]()
            ttnn.synchronize_device(dev)

    held = clk.force(MHZ, [node])
    sampler = clk.Sampler(node)
    time.sleep(0.3)

    res = {n: [] for n in ORDER}
    t_start = time.perf_counter()
    for rep in range(REPS):
        for name in ORDER:
            if name == "silu":
                ttnn.copy(x0, x)          # fresh values, outside the timed region
            ttnn.synchronize_device(dev)
            fn = ARMS[name]
            t0 = time.perf_counter()
            for _ in range(CALLS):
                fn()
            ttnn.synchronize_device(dev)
            res[name].append((time.perf_counter() - t0) / CALLS * 1e3)
        print(f"rep {rep}: " + "  ".join(f"{n}={res[n][-1]:.4f}" for n in ORDER), flush=True)
    t_end = time.perf_counter()

    aiclk = sampler.stop()
    clk.release()
    ttnn.close_device(dev)

    def summary(n):
        v = res[n]
        return {"ms_per_call_median": statistics.median(v), "ms_per_call_min": min(v),
                "ms_per_call_max": max(v), "reps": len(v), "calls_per_rep": CALLS}

    c = counts()
    med = {n: statistics.median(res[n]) for n in ORDER}
    tax = med["fc1_fused"] - med["fc1_plain"]
    out = {
        "card_node": node, "clock_forced_mhz": MHZ, "clock_held_nodes": held,
        "aiclk_during_session": aiclk, "timed_seconds": round(t_end - t_start, 2),
        "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {n: summary(n) for n in ORDER}, "counts": c,
        "derived": {
            "silu_ms": med["silu"],
            "tax_ms_per_call": tax,
            "tax_seconds_per_fold": tax * CALLS_PER_FOLD / 1e3,
            "op_level_win_ms_per_call": med["fc1_fused"] - med["fc1_unfused"],
            "op_level_win_seconds_per_fold": (med["fc1_fused"] - med["fc1_unfused"]) * CALLS_PER_FOLD / 1e3,
            "proxy_win_seconds_from_silu_ms": (tax - med["silu"]) * CALLS_PER_FOLD / 1e3,
            "aa_ratio_fused_over_fused_aa": med["fc1_fused"] / med["fc1_fused_aa"],
            "aa_abs_ms": abs(med["fc1_fused"] - med["fc1_fused_aa"]),
            "aa_seconds_per_fold": abs(med["fc1_fused"] - med["fc1_fused_aa"]) * CALLS_PER_FOLD / 1e3,
            "dispatch_floor_ms": med["dispatch_floor"],
            "hostloop_ms": med["hostloop"],
            "silu_gbps": c["silu_bytes_logical"] / (med["silu"] / 1e3) / 1e9,
            "fc1_fused_tflops": c["fc1_flops_logical"] / (med["fc1_fused"] / 1e3) / 1e12,
            "fc1_plain_tflops": c["fc1_flops_logical"] / (med["fc1_plain"] / 1e3) / 1e12,
            "kill_a_silu_ge_epilogue": med["silu"] >= tax,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("DERIVED " + json.dumps(out["derived"]))
    print("COUNTS " + json.dumps(c))
    print("AICLK " + json.dumps(aiclk))


if __name__ == "__main__":
    main()
