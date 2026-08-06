#!/usr/bin/env python3
"""Characterise `ttnn.graph` capture as a profiling instrument on Blackhole.

`ttnn.graph.begin_graph_capture` / `end_graph_capture` is the host-side op-graph
recorder. It needs no Tracy build and no device profiler, so it is the obvious
reach for "how many ops does my model dispatch?" — but it has three traps that
this probe measures rather than assumes:

  overhead  how much it perturbs the thing it measures (vs bare, both synced)
  scale     host RSS growth per captured op — its scale wall is host RAM, and
            it is reached silently (OOM), not with a loud error
  duration  `duration_ns` on a function_end node is the INSTRUMENTED HOST wall
            time, not device kernel time and not enqueue time — a third number
  l1        `extract_peak_L1_memory_usage` returns 0 for a program-cached (warm)
            op, because circular-buffer allocation is only recorded on the
            cache-miss path. L1 must be captured COLD; timing must be warm.

Run each leg separately; `--only l1` and `--only duration` are cheap, `--only
scale` allocates ~9 GB of host RAM at the top of the sweep.

    export TT_VISIBLE_DEVICES=0
    python3 scripts/profiling/graph_capture_probe.py --only overhead
"""

import argparse
import resource
import time

import torch

import ttnn


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _capture(fn):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    return ttnn.graph.end_graph_capture()


def leg_overhead(device, iters):
    """Bare vs captured, same loop, both synced before the clock stops."""
    x = ttnn.from_torch(
        torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(10):
        ttnn.deallocate(ttnn.add(x, x))
    ttnn.synchronize_device(device)

    def loop():
        for _ in range(iters):
            ttnn.deallocate(ttnn.add(x, x))

    t0 = time.perf_counter()
    loop()
    ttnn.synchronize_device(device)
    bare = (time.perf_counter() - t0) / iters * 1e6

    t0 = time.perf_counter()
    graph = _capture(loop)
    ttnn.synchronize_device(device)
    cap = (time.perf_counter() - t0) / iters * 1e6

    print(
        f"OVERHEAD iters={iters} bare={bare:.2f}us captured={cap:.2f}us "
        f"overhead=+{100 * (cap / bare - 1):.0f}% nodes={len(graph)} "
        f"nodes_per_op={len(graph) / iters:.1f}"
    )


def leg_scale(device, sizes):
    """Host RSS per captured op. This is graph capture's real scale wall."""
    x = ttnn.from_torch(
        torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(10):
        ttnn.deallocate(ttnn.add(x, x))
    ttnn.synchronize_device(device)
    for n in sizes:
        before = rss_mb()
        graph = _capture(lambda: [ttnn.deallocate(ttnn.add(x, x)) for _ in range(n)])
        ttnn.synchronize_device(device)
        after = rss_mb()
        print(
            f"SCALE ops={n} nodes={len(graph)} rss_before={before:.0f}MB "
            f"rss_after={after:.0f}MB delta={after - before:.0f}MB "
            f"bytes_per_op={(after - before) * 1e6 / n:.0f}",
            flush=True,
        )
        del graph


def leg_duration(device, n):
    """`duration_ns` vs the honest synced host time vs the unsynced enqueue."""
    x = ttnn.from_torch(
        torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for _ in range(3):
        ttnn.deallocate(ttnn.matmul(x, x))
    ttnn.synchronize_device(device)

    def loop():
        for _ in range(10):
            ttnn.deallocate(ttnn.matmul(x, x))

    t0 = time.perf_counter()
    loop()
    ttnn.synchronize_device(device)
    honest = (time.perf_counter() - t0) / 10 * 1e6

    t0 = time.perf_counter()
    loop()
    unsynced = (time.perf_counter() - t0) / 10 * 1e6
    ttnn.synchronize_device(device)

    graph = _capture(loop)
    ds = [
        node["duration_ns"]
        for node in graph
        if node.get("node_type") == "function_end"
        and (node.get("params") or {}).get("name") == "ttnn.matmul"
    ]
    reported = sum(ds) / len(ds) / 1000
    print(
        f"DURATION matmul {n}^3: honest_synced={honest:.1f}us "
        f"unsynced_enqueue={unsynced:.1f}us (={honest / unsynced:.2f}x underreport) "
        f"graph_duration_ns={reported:.1f}us (={reported / honest:.2f}x over honest)"
    )


def leg_l1(device):
    """peak_L1 is recorded only on the program-cache-miss path -> capture COLD."""
    for n in (640, 704):
        x = ttnn.from_torch(
            torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for tag in ("COLD", "WARM"):
            graph = _capture(lambda: ttnn.deallocate(ttnn.matmul(x, x)))
            cb = sum(
                1 for node in graph if "circular_buffer_allocate" in str(node.get("node_type"))
            )
            print(
                f"L1 shape={n} {tag}: peak_L1={ttnn.graph.extract_peak_L1_memory_usage(graph)} "
                f"cb_alloc_nodes={cb} nodes={len(graph)}"
            )
            if tag == "COLD":  # warm it, so the second capture is a cache hit
                for _ in range(5):
                    ttnn.deallocate(ttnn.matmul(x, x))
                ttnn.synchronize_device(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--only", default="overhead", choices=("overhead", "scale", "duration", "l1"))
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--mm_size", type=int, default=2048)
    args = ap.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        if args.only == "overhead":
            leg_overhead(device, args.iters)
        elif args.only == "scale":
            leg_scale(device, (1000, 10000, 100000))
        elif args.only == "duration":
            leg_duration(device, args.mm_size)
        else:
            leg_l1(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
