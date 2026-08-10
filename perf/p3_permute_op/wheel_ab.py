#!/usr/bin/env python3
"""Card or wheel? The same kernel, the same qb1 card 0, run under 0.67.4 and under 0.68.0.

X6 measured this op at 1.082x against stock `ttnn.permute` at N=298 on L1, on qb2 card 2 at ttnn
0.68.0. The qb1 / 0.67.4 probe measures 0.832x. Two variables moved at once — the card and the wheel
— so this script holds the card fixed and moves only the wheel: qb1 has a 0.68.0 wheel in
`/home/ttuser/tt-boltz2/env` next to tt-bio's own 0.67.4.

Deliberately imports only `ttnn` and `tt_bio/reblock_permute.py` (loaded by path, so no tt-bio
dependency has to resolve under the foreign env) and opens the device with `ttnn.open_device`, so the
two runs differ in nothing but the wheel.

    TT_VISIBLE_DEVICES=0 <env>/bin/python3 perf/p3_permute_op/wheel_ab.py --out <f>.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import torch, ttnn

try:
    from importlib.metadata import version
    WHEEL = version("ttnn")
except Exception:
    WHEEL = "unknown"


def load_rp():
    spec = importlib.util.spec_from_file_location("rp", REPO / "tt_bio" / "reblock_permute.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.set_enabled(True)
    return m


def timeit(device, fn, reps=21, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def throughput_us(device, fn, k=40, warmup=5):
    """Per-call cost with K calls issued back to back and ONE sync at the end.

    This is the production-relevant rate: inside the fold, calls are enqueued back to back, so a
    host cost below the device time hides behind the previous call's execution. `timeit` above syncs
    per call and therefore charges host and device serially.
    """
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=32,
                    help="chunk channel width. 64 is what a 13x10-grid 298 aa fold constructs; "
                         "32 is what an 11x10 grid does, and what X6 measured.")
    a = ap.parse_args()

    RP = load_rp()
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    R = {"wheel": WHEEL, "host": "qb1", "card": 0, "device_grid": [int(g.x), int(g.y)],
         "per_core_l1_unreserved_B": int(ttnn.get_max_worker_l1_unreserved_size()), "rows": []}
    print("wheel", WHEEL, "grid", R["device_grid"], flush=True)

    MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
    C = a.channels
    R["channels"] = C
    for N in (298, 320):
        ref = torch.randn(1, N, N, C, dtype=torch.bfloat16)
        gold = ref.permute(0, 3, 1, 2).contiguous()
        nbytes = N * N * C * 2
        for where in ("l1", "dram"):
            mc = MC[where]
            x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                                memory_config=mc)
            r = RP.reblock_permute(x, mc, device)
            eq = bool(torch.equal(ttnn.to_torch(r), gold))
            ttnn.deallocate(r)

            wired = lambda x=x, mc=mc: ttnn.deallocate(RP.reblock_permute(x, mc, device))
            stock = lambda x=x, mc=mc: ttnn.deallocate(
                ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))
            clone = lambda x=x, mc=mc: ttnn.deallocate(ttnn.clone(x, memory_config=mc))

            row = {"N": N, "C": C, "buf": where, "torch_equal": eq,
                   "wired_synced_us": round(timeit(device, wired), 2),
                   "stock_synced_us": round(timeit(device, stock), 2),
                   "wired_thru_us": round(throughput_us(device, wired), 2),
                   "stock_thru_us": round(throughput_us(device, stock), 2),
                   "clone_same_buf_us": round(timeit(device, clone), 2)}
            row["ratio_synced"] = round(row["stock_synced_us"] / row["wired_synced_us"], 3)
            row["ratio_thru"] = round(row["stock_thru_us"] / row["wired_thru_us"], 3)
            row["clone_GBs"] = round(nbytes / row["clone_same_buf_us"] / 1e3, 1)
            row["wired_over_clone"] = round(row["wired_synced_us"] / row["clone_same_buf_us"], 2)
            R["rows"].append(row)
            print(row, flush=True)
            ttnn.deallocate(x)

    # Host cost on the cache-hit path, on this wheel's bindings.
    ref = torch.randn(1, 298, 298, C, dtype=torch.bfloat16)
    x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, 298, 298]), ttnn.bfloat16,
                                            ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
        entry = RP._prepare(x, out, device)
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = [x.buffer_address()]
        pd.kernels[1].common_runtime_args = [out.buffer_address()]
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    R["host_cached_per_call_us"] = round(ts[len(ts) // 2], 2)
    print("host_cached_per_call_us", R["host_cached_per_call_us"], flush=True)

    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
