#!/usr/bin/env python3
"""The per-op launch floor on Blackhole p300c, one op class at a time.

`roof-difftx-arith-efficiency` fit this for `ttnn.layer_norm` alone, on pc's p150a and on a
Wormhole Galaxy: 12.77 us fixed + 3.69 us per 512x768 block (BH) and 20.68 + 8.44 (WH). It also
settled WHAT the cost is -- a device-side program launch, not host dispatch: the same arms replayed
from a captured trace with the host removed returned 16.46 us against a synced 16.13. So the term
is real, it is serial with the rest of the op stream, and no roofline in this campaign carries it.

This sweeps the same construction across every op class the 512 aa fold actually launches, on the
part the floor of record is stated on (qb2 p300c), and fits

    t(x) = fixed + slope * x        x = the op's OUTPUT tile count

by least squares over a size ladder. `fixed` is the y-intercept: what an op of this class costs
with no work in it. That is the third term; `slope` is work and is already priced by the traffic
and arithmetic terms, so it never enters the join.

The floor this file publishes per class is

    launch_floor = min( fit intercept if it is positive , the cheapest point on the ladder )

which is the intercept wherever the ladder is monotone and the fit is clean, and the cheapest
measured member of the class wherever it is not. `ttnn.matmul` is the case that needs the second
branch: it is NOT monotone in x, a 32-row matmul costs 21.0 us against 11.1 us for a 128-row one,
so a line through it has a meaningless intercept. Taking the cheapest measured point instead keeps
the term a floor.

Timing is `roof_difftx/dit_layer.py`'s: warm the program cache, then enqueue `reps` calls,
synchronize once, divide, and take the min over `blocks` blocks. One sync per block is what a fold
pays -- ops go out back to back -- so this is a steady-state per-op cost, not a one-op latency.
The blocks are INTERLEAVED across every ladder point rather than run arm by arm, because the first
cut of this sweep ran them arm by arm and its A/A read 29.9 % on one arm.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

DIM = 768
LADDER = [32, 64, 128, 256, 512, 1024]


def tiles(shape) -> int:
    n = 1
    for d in shape[:-2]:
        n *= d
    return n * ceil(shape[-2] / 32) * ceil(shape[-1] / 32)


def build(dev, kc):
    """name -> (rows -> (callable, x_tiles, in_place)). One entry per op class the fold launches."""

    def t(shape, sc=1.0, layout=ttnn.TILE_LAYOUT, dt=torch.bfloat16):
        return ttnn.from_torch(torch.randn(*shape, dtype=dt) * sc, layout=layout, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mk(fn, out_shape, *in_shapes, in_place=False):
        """x is the op's OUTPUT tile count; an in-place op has none, so it takes its input's.

        Not the max over every operand: `ttnn.linear`'s 768x768 weight is 576 tiles and would
        clamp the whole ladder below 512 rows, leaving four of six points at the same x and a
        fit with no leverage. The join uses the same rule, so the sweep's x and the fold op's x
        are one definition.
        """
        x = tiles(out_shape) if out_shape else max(tiles(s) for s in in_shapes)
        return fn, x, in_place

    arms = {}

    def reg(name):
        def deco(f):
            arms[name] = f
            return f
        return deco

    @reg("layer_norm")
    def _ln(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=kc),
                  (1, R, DIM))

    @reg("layer_norm_w")
    def _lnw(R):
        x = t((1, R, DIM))
        w = t((1, 1, DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        b = t((1, 1, DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        return mk(lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5,
                                          compute_kernel_config=kc), (1, R, DIM))

    @reg("add")
    def _add(R):
        x, y = t((1, R, DIM)), t((1, R, DIM))
        return mk(lambda: ttnn.add(x, y), (1, R, DIM))

    @reg("add_")
    def _addi(R):
        x, y = t((1, R, DIM)), t((1, R, DIM))
        return mk(lambda: ttnn.add_(x, y), None, (1, R, DIM), in_place=True)

    @reg("multiply")
    def _mul(R):
        x, y = t((1, R, DIM)), t((1, R, DIM))
        return mk(lambda: ttnn.multiply(x, y), (1, R, DIM))

    @reg("multiply_")
    def _muli(R):
        x, y = t((1, R, DIM)), t((1, R, DIM))
        return mk(lambda: ttnn.multiply_(x, y), None, (1, R, DIM), in_place=True)

    @reg("linear")
    def _lin(R):
        x, w = t((1, R, DIM)), t((DIM, DIM))
        return mk(lambda: ttnn.linear(x, w, compute_kernel_config=kc,
                                      core_grid=T.CORE_GRID_MAIN), (1, R, DIM))

    @reg("matmul")
    def _mm(R):
        x, w = t((1, R, DIM)), t((DIM, DIM))
        return mk(lambda: ttnn.matmul(x, w, compute_kernel_config=kc,
                                      core_grid=T.CORE_GRID_MAIN), (1, R, DIM))

    @reg("permute")
    def _perm(R):
        x = t((1, 16, R, 48))
        return mk(lambda: ttnn.permute(x, (0, 1, 3, 2)), (1, 16, 48, R))

    @reg("reshape")
    def _resh(R):
        x = t((1, 16, 48, R))
        return mk(lambda: ttnn.reshape(x, (1, 768, R)), (1, 768, R))

    @reg("transpose")
    def _tr(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.transpose(x, -2, -1), (1, DIM, R))

    @reg("to_layout")
    def _tl(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, R, DIM))

    @reg("to_memory_config_l1")
    def _tmc(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), (1, R, DIM))

    @reg("to_memory_config_noop")
    def _tmcn(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, R, DIM),
                  in_place=True)

    @reg("slice")
    def _sl(R):
        x = t((1, 16, R, 64))
        return mk(lambda: ttnn.slice(x, (0, 0, 0, 0), (1, 16, R, 32)), (1, 16, R, 32))

    @reg("concat")
    def _cat(R):
        x, y = t((1, R, DIM)), t((1, R, DIM))
        return mk(lambda: ttnn.concat([x, y], dim=-1), (1, R, 2 * DIM))

    @reg("softmax")
    def _sm(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.softmax(x, dim=-1, compute_kernel_config=kc), (1, R, DIM))

    @reg("pad")
    def _pad(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.pad(x, ((0, 0), (0, 32), (0, 0)), 0.0), (1, R + 32, DIM))

    @reg("sdpa")
    def _sdpa(R):
        q, k, v = t((1, 16, R, 64)), t((1, 16, R, 64)), t((1, 16, R, 64))
        return mk(lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False), (1, 16, R, 64))

    @reg("nlp_create_qkv_heads")
    def _qkv(R):
        x = t((1, 1, R, 3072))
        return mk(lambda: ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=16, num_kv_heads=16, transpose_k_heads=False), (3, 16, R, 64))

    @reg("nlp_concat_heads")
    def _nch(R):
        x = t((1, 16, R, 64))
        return mk(lambda: ttnn.experimental.nlp_concat_heads(x), (1, 1, R, 1024))

    @reg("chunk")
    def _ch(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.chunk(x, 2, dim=-1), (1, R, DIM))

    @reg("cos")
    def _cos(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.cos(x), (1, R, DIM))

    # host-side metadata, no device program expected. Measured to PROVE the 148k calls the fold
    # makes to them carry no launch floor, rather than assuming it.
    @reg("unsqueeze")
    def _un(R):
        x = t((1, R, DIM))
        return mk(lambda: ttnn.unsqueeze(x, 0), None, (1, R, DIM), in_place=True)

    @reg("squeeze")
    def _sq(R):
        x = t((1, 1, R, DIM))
        return mk(lambda: ttnn.squeeze(x, 0), None, (1, 1, R, DIM), in_place=True)

    @reg("getitem")
    def _gi(R):
        x = t((1, R, DIM))
        return mk(lambda: x[0], (R, DIM))

    return arms


def warm(dev, fn, in_place):
    for _ in range(3):
        o = fn()
        if not in_place:
            _drop(o)
    ttnn.synchronize_device(dev)


def one_block(dev, fn, reps, in_place):
    """`reps` enqueues, one barrier, divided. What a fold pays: ops go out back to back."""
    held = []
    t0 = time.perf_counter()
    for _ in range(reps):
        o = fn()
        if not in_place:
            held.append(o)
            if len(held) > 4:
                _drop(held.pop(0))
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    for h in held:
        _drop(h)
    return dt * 1e6


def time_arm(dev, fn, reps, blocks, in_place):
    warm(dev, fn, in_place)
    return min(one_block(dev, fn, reps, in_place) for _ in range(blocks))


def _drop(o):
    for x in (o if isinstance(o, (list, tuple)) else [o]):
        try:
            ttnn.deallocate(x)
        except Exception:                                                     # noqa: BLE001
            pass


def trace_point(dev, fn, reps, blocks, in_place):
    """Device time per op with the host removed: capture `reps` enqueues once, replay, divide.

    This is the reading the floor is published on. The synced reading is max(host, device) and for
    the cheap eltwise classes the host half is the larger one -- `ttnn.add` at 512 rows reads 14.25
    us synced against 13.76 host and 8.97 traced -- so publishing the synced number there would put
    host dispatch into a roofline. Host dispatch is hidden under device time in a real fold
    (`roof-difftx` measured a whole DiT layer at 1138.5 us synced, 1143.2 traced, host 528.5).
    """
    warm(dev, fn, in_place)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    held = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    best = None
    for _ in range(blocks):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / reps
        best = dt if best is None else min(best, dt)
    ttnn.release_trace(dev, tid)
    if not in_place:
        for h in held:
            _drop(h)
    ttnn.synchronize_device(dev)
    return best * 1e6


def probe(dev, arms, names, R, reps, blocks):
    """t_sync / t_host / t_trace for one size, `roof_difftx/dispatch_probe.py`'s three readings.

    t_sync  enqueue `reps`, synchronize, divide   = max(host, device) + tail
    t_host  the same enqueues, clock stopped BEFORE the barrier = host dispatch alone
    t_trace replay a captured trace of `reps` ops = the device alone, host removed

    The floor above is only a DEVICE launch floor if t_trace tracks t_sync. If t_host tracked it
    instead the number would be host dispatch, which a captured trace removes, and it would have no
    business in a roofline. `roof-difftx` ran this on p150a for one op; this runs it on the part of
    record across every class.
    """
    out = {}
    for name in names:
        try:
            fn, x, ip = arms[name](R)
        except Exception as e:                                                # noqa: BLE001
            out[name] = {"error": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:120])}
            continue
        warm(dev, fn, ip)
        row = {"x_tiles": x, "t_sync_us": min(one_block(dev, fn, reps, ip)
                                              for _ in range(blocks))}
        best = None
        for _ in range(blocks):
            held = []
            t0 = time.perf_counter()
            for _ in range(reps):
                o = fn()
                if not ip:
                    held.append(o)
                    if len(held) > 4:
                        _drop(held.pop(0))
            dt = (time.perf_counter() - t0) / reps
            ttnn.synchronize_device(dev)
            for h in held:
                _drop(h)
            best = dt if best is None else min(best, dt)
        row["t_host_us"] = best * 1e6
        try:
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            held = [fn() for _ in range(reps)]
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            best = None
            for _ in range(blocks):
                t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
                best = dt if best is None else min(best, dt)
            row["t_trace_us"] = best * 1e6
            ttnn.release_trace(dev, tid)
            if not ip:
                for h in held:
                    _drop(h)
        except Exception as e:                                                # noqa: BLE001
            row["t_trace_us"] = None
            row["trace_error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:120])
        ttnn.synchronize_device(dev)
        out[name] = row
        tr = row["t_trace_us"]
        print("  %-22s sync %7.2f  host %7.2f  trace %s"
              % (name, row["t_sync_us"], row["t_host_us"],
                 ("%7.2f us" % tr) if tr else row.get("trace_error", "-")[:70]), flush=True)
    return out


def fit(xs, ys):
    """least squares y = a + b x; returns (a, b, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def roofs(dev, kc, blocks):
    """the part's own cube and starved-add bandwidth this session, so the row is not asserted."""
    out = {}
    a = ttnn.from_torch(torch.randn(4096, 4096, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(4096, 4096, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    us = time_arm(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=kc,
                                           core_grid=T.CORE_GRID_MAIN), 3, blocks, False)
    out["cube4096_TFLOPs"] = 2 * 4096 ** 3 / (us * 1e-6) / 1e12
    ttnn.deallocate(a); ttnn.deallocate(b)
    c = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    d = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    us = time_arm(dev, lambda: ttnn.add(c, d), 3, blocks, False)
    out["add8192_GBps"] = 3 * 8192 * 8192 * 2 / (us * 1e-6) / 1e9
    ttnn.deallocate(c); ttnn.deallocate(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "launch_sweep.json")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=6, help="interleaved passes over every point")
    ap.add_argument("--only", default=None, help="comma-separated arm names")
    ap.add_argument("--probe-rows", type=int, default=512)
    ap.add_argument("--mode", choices=("sync", "trace"), default="sync")
    ap.add_argument("--skip", default=None, help="comma-separated arm names to leave out")
    ap.add_argument("--trace-region", type=int, default=1 << 29)
    a = ap.parse_args()

    dev = T.get_device(trace_region_size=a.trace_region)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {"host": platform.node(), "arch": str(dev.arch()),
           "grid": [dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y],
           "core_grid_main": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y],
           "reps": a.reps, "blocks": a.blocks, "ladder": LADDER,
           "loadavg_before": open("/proc/loadavg").read().split()[:3],
           "x_is": "the op's output tile count (an in-place op: its largest input's)",

           "rows": {}, "fits": {}, "refused": {}}
    out["reading"] = ("device only, trace replay" if a.mode == "trace"
                      else "max(host dispatch, device), one barrier per block")
    out["roofs"] = roofs(dev, kc, min(a.blocks, 5))
    print("cube4096 %.2f TFLOP/s   add8192 %.1f GB/s"
          % (out["roofs"]["cube4096_TFLOPs"], out["roofs"]["add8192_GBps"]), flush=True)

    arms = build(dev, kc)
    want = [n for n in (a.only.split(",") if a.only else list(arms)) if n in arms]
    if a.skip:
        want = [n for n in want if n not in set(a.skip.split(","))]

    if a.mode == "trace":
        # one arm at a time, JSON written after each, because a trace capture can wedge the card
        # and a wedge is not an exception -- `ttnn.transpose` wedged this sweep at 100 % CPU in
        # futex_do_wait and had to be killed by pid. Incremental writes keep the arms already done.
        for name in want:
            pts = []
            for R in LADDER:
                try:
                    fn, x, ip = arms[name](R)
                    us = trace_point(dev, fn, a.reps, a.blocks, ip)
                    pts.append({"rows": R, "x_tiles": x, "us": us})
                except Exception as e:                                        # noqa: BLE001
                    out["refused"]["%s@%d" % (name, R)] = "%s: %s" % (
                        type(e).__name__, str(e).splitlines()[0][:160])
            out["rows"][name] = pts
            if len(pts) >= 3:
                a0, b0, r2 = fit([q["x_tiles"] for q in pts], [q["us"] for q in pts])
                lo = min(q["us"] for q in pts)
                floor_us = min(a0 if a0 > 0 else float("inf"), lo)
                out["fits"][name] = {"fixed_us": a0, "slope_us_per_tile": b0, "r2": r2,
                                     "n": len(pts), "min_measured_us": lo,
                                     "launch_floor_us": floor_us,
                                     "floor_from": "fit intercept" if floor_us == a0
                                                   else "measured min"}
                print("%-22s fixed %8.2f  slope %7.4f  r2 %.4f  min %6.2f  FLOOR %6.2f us  | %s"
                      % (name, a0, b0, r2, lo, floor_us,
                         " ".join("%.1f" % q["us"] for q in pts)), flush=True)
            else:
                print("%-22s REFUSED %s" % (name, [k for k in out["refused"]
                                                   if k.startswith(name + "@")]), flush=True)
            a.out.write_text(json.dumps(out, indent=1))
        out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
        out["roofs_after"] = roofs(dev, kc, 5)
        out["AA_pct"] = {"cube4096": 100 * abs(out["roofs_after"]["cube4096_TFLOPs"]
                                               - out["roofs"]["cube4096_TFLOPs"])
                         / out["roofs"]["cube4096_TFLOPs"],
                         "add8192": 100 * abs(out["roofs_after"]["add8192_GBps"]
                                              - out["roofs"]["add8192_GBps"])
                         / out["roofs"]["add8192_GBps"]}
        print("A/A cube %.2f %%, add8192 %.2f %%   cube %.2f -> %.2f TFLOP/s"
              % (out["AA_pct"]["cube4096"], out["AA_pct"]["add8192"],
                 out["roofs"]["cube4096_TFLOPs"], out["roofs_after"]["cube4096_TFLOPs"]),
              flush=True)
        a.out.write_text(json.dumps(out, indent=1))
        print("-> %s" % a.out, flush=True)
        return 0

    # Build and warm EVERY ladder point    print("device vs host, %d rows:" % a.probe_rows, flush=True)
    out["probe_rows"] = a.probe_rows
    out["probe"] = probe(dev, arms, [n for n in want if n in arms], a.probe_rows,
                         a.reps, min(a.blocks, 5))

    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    out["roofs_after"] = roofs(dev, kc, a.blocks)
    print("cube4096 %.2f -> %.2f TFLOP/s   add8192 %.1f -> %.1f GB/s"
          % (out["roofs"]["cube4096_TFLOPs"], out["roofs_after"]["cube4096_TFLOPs"],
             out["roofs"]["add8192_GBps"], out["roofs_after"]["add8192_GBps"]), flush=True)
    out["AA_pct"] = {
        "cube4096": 100 * abs(out["roofs_after"]["cube4096_TFLOPs"]
                              - out["roofs"]["cube4096_TFLOPs"]) / out["roofs"]["cube4096_TFLOPs"],
        "add8192": 100 * abs(out["roofs_after"]["add8192_GBps"]
                             - out["roofs"]["add8192_GBps"]) / out["roofs"]["add8192_GBps"]}
    print("A/A cube %.2f %%, add8192 %.2f %%"
          % (out["AA_pct"]["cube4096"], out["AA_pct"]["add8192"]), flush=True)

    a.out.write_text(json.dumps(out, indent=1))
    print("-> %s" % a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
