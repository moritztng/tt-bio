#!/usr/bin/env python3
"""Does an L1 destination move the diffusion transformer's 768-dim projections? Ablation, not a counter.

`shape_gap.py` found the mass: four shapes inside `DiffusionTransformerLayer` carry 1.4551 s of the
matmul class's 3.9459 s at 512 aa, the biggest of them (1,512,768)x(768,768) at 38,600 calls and
0.7976 s, and the capture's own memory columns read DRAM_INTERLEAVED on all three operands of every
one of those calls. The same module's `Transition` already passes L1_MEMORY_CONFIG and its shapes
attain 0.467-0.502 of their roofs where the all-DRAM one attains 0.307.

The instrument is the ablation c13 used, for the same reason: four campaigns here were derailed by
counters returning confident wrong numbers, and no counter is read anywhere in this file. Each arm
removes exactly one term and the delta prices it.

  ship      the fold's configuration, every operand DRAM       -> the base the fold pays
  grid      CORE_GRID_MAIN passed instead of ttnn's default    -> prices our routing vs ttnn's
  out_l1    result in L1                                       -> prices the DRAM WRITE term
  in_l1     activation in L1                                   -> prices the DRAM READ term
  both_l1   activation and result in L1                        -> prices the whole DRAM term

KNOWN-ANSWER CONTROLS in the same session, before any arm is believed: an 8192^3 cube whose FLOP
count is exactly 1,099,511,627,776 counted twice by independent routes, under the config the arms
themselves run (HiFi4 / fp32_dest_acc_en / packer_l1_acc), pre-registered at 120.06 TFLOP/s in
[112, 128]; and a starved 8192^2 add of exactly 402,653,184 bytes, pre-registered at 433.7 GB/s in
[400, 460]. Bytes are a closed form over dense shapes touched once, never a counter. Every arm
carries an A/A twin and a twin that spreads more than 3 % refuses the session.

Clock is forced by THIS process over tt-kmd's ARC queue, because a helper that both forces and
samples can be taken away with the device (c13 session s2 lost its force 21 s into a 4-minute run
and the record still read 99.8 % at target).
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import statistics as st
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

_IOC = (0xFA << 8) | 17
_POST, _POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33


def _smc(fd, msg_type, *args):
    msg = [msg_type] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, _IOC, struct.pack("=IIII8I", 48, _POST, 0, 0, *msg))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        buf = bytearray(struct.pack("=IIII8I", 48, _POLL, 0, 0, *([0] * 8)))
        try:
            fcntl.ioctl(fd, _IOC, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005)
                continue
            raise
        return struct.unpack("=IIII8I", bytes(buf))[4] & 0xFF
    raise TimeoutError("no ARC response")


def read_aiclk(node):
    try:
        return int(Path("/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % node).read_text())
    except OSError as e:
        return "ERR:%s" % e.errno


sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# The four shapes, with the fold call counts and in-situ seconds shape_gap_insitu.json measured.
SHAPES = {
    "qkvo": dict(b=1, m=512, k=768, n=768, calls=38600, insitu_s=0.79761, insitu_us=20.66),
    "fc12": dict(b=1, m=512, k=768, n=1536, calls=15200, insitu_s=0.34394, insitu_us=22.63),
    "fc3": dict(b=1, m=512, k=1536, n=768, calls=5200, insitu_s=0.12107, insitu_us=23.28),
    "wide": dict(b=1, m=512, k=768, n=3072, calls=4800, insitu_s=0.19248, insitu_us=40.10),
}


def flops(b, m, k, n):
    return 2.0 * b * m * k * n


def tile_flops(b, m, k, n):
    assert m % 32 == 0 and k % 32 == 0 and n % 32 == 0
    return float(b) * (m // 32) * (k // 32) * (n // 32) * 2 * 32 ** 3


def min_bytes(b, m, k, n, elem=2):
    """Compulsory traffic: activation once, shared weight once, result once."""
    return float(elem) * (b * m * k + k * n + b * m * n)


def timed(fn, device, reps, warm):
    out = []
    for i in range(reps + warm):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
        if i >= warm:
            out.append((t1 - t0) * 1e3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--skip-cube", action="store_true")
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T

    clk_fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    before = read_aiclk(a.node)
    status = "0x%02X" % _smc(clk_fd, FORCE_AICLK, a.clock)
    print("node%d FORCE_AICLK(%d) status=%s before=%s" % (a.node, a.clock, status, before),
          flush=True)

    def _release():
        try:
            _smc(clk_fd, FORCE_AICLK, 0)
            print("node%d clock force released" % a.node, flush=True)
        except Exception as e:                                                # noqa: BLE001
            print("node%d RELEASE FAILED %r" % (a.node, e), flush=True)

    atexit.register(_release)
    t0 = time.time()
    while read_aiclk(a.node) != a.clock and time.time() - t0 < 10.0:
        time.sleep(0.01)
    if read_aiclk(a.node) != a.clock:
        print("REFUSING: node%d at %s MHz, not %d" % (a.node, read_aiclk(a.node), a.clock))
        return 2
    print("node%d reached %d MHz after %.0f ms" % (a.node, a.clock, 1e3 * (time.time() - t0)),
          flush=True)
    t_open = time.time()

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    grid = T.CORE_GRID_MAIN
    print("grid %dx%d = %d cores, CORE_GRID_MAIN=%r" % (g.x, g.y, g.x * g.y, grid), flush=True)
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)

    res = {"env": {"node": a.node, "clock": a.clock, "grid": [g.x, g.y], "tag": a.tag,
                   "reps": a.reps, "warm": a.warm, "t_open": t_open,
                   "ttnn": ttnn.__file__, "arch": str(device.arch())},
           "controls": {}, "arms": {}}

    if not a.skip_cube:
        n = 8192
        f1, f2 = flops(1, n, n, n), tile_flops(1, n, n, n)
        assert f1 == f2 == 2 * n ** 3, (f1, f2)
        A = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=DRAM)
        ms = timed(lambda: ttnn.matmul(A, B, compute_kernel_config=KC, memory_config=DRAM),
                   device, 5, 2)
        tf = f1 / 1e12 / (st.median(ms) / 1e3)
        res["controls"]["cube8192"] = {"flops_route1": f1, "flops_route2": f2,
                                       "ms": [round(x, 4) for x in ms],
                                       "tflops": round(tf, 3), "band": [112, 128],
                                       "pass": 112 <= tf <= 128}
        print("CONTROL cube8192 %.3f TFLOP/s  pass=%s"
              % (tf, res["controls"]["cube8192"]["pass"]), flush=True)
        ttnn.deallocate(A)
        ttnn.deallocate(B)
        nb = 3 * 8192 * 8192 * 2
        assert nb == 402653184, nb
        C = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        D = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        ms = timed(lambda: ttnn.add(C, D, memory_config=DRAM), device, 7, 2)
        gbs = nb / 1e9 / (st.median(ms) / 1e3)
        res["controls"]["dram_add"] = {"bytes": nb, "ms": [round(x, 4) for x in ms],
                                       "gbs": round(gbs, 2), "band": [400, 460],
                                       "pass": 400 <= gbs <= 460}
        print("CONTROL dram_add %.2f GB/s  pass=%s"
              % (gbs, res["controls"]["dram_add"]["pass"]), flush=True)
        ttnn.deallocate(C)
        ttnn.deallocate(D)

    arms = []
    for name, s in SHAPES.items():
        b, m, k, n = s["b"], s["m"], s["k"], s["n"]
        x_d = ttnn.from_torch(torch.randn(b, m, k, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=DRAM)
        x_l = ttnn.from_torch(torch.randn(b, m, k, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=L1)
        w = ttnn.from_torch(torch.randn(k, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=DRAM)

        def mk(xt, mc, cg, w=w):
            def f():
                kw = {"compute_kernel_config": KC, "memory_config": mc, "dtype": ttnn.bfloat16}
                if cg is not None:
                    kw["core_grid"] = cg
                return ttnn.linear(xt, w, **kw)
            return f

        for arm, xt, mc, cg in (("ship", x_d, DRAM, None),
                                ("ship_aa", x_d, DRAM, None),
                                ("grid", x_d, DRAM, grid),
                                ("out_l1", x_d, L1, None),
                                ("in_l1", x_l, DRAM, None),
                                ("both_l1", x_l, L1, None),
                                ("both_l1_aa", x_l, L1, None)):
            arms.append(("%s/%s" % (name, arm), mk(xt, mc, cg), name))

    for fn_key, fn, _ in arms:
        timed(fn, device, 0, a.warm)
    samples = {k: [] for k, _, _ in arms}
    for r in range(a.reps):
        for fn_key, fn, _ in arms:
            samples[fn_key] += timed(fn, device, 1, 0)

    clk_end = read_aiclk(a.node)
    for fn_key, _, sname in arms:
        s = SHAPES[sname]
        ms = samples[fn_key]
        med = st.median(ms)
        f = flops(s["b"], s["m"], s["k"], s["n"])
        res["arms"][fn_key] = {
            "shape": [s["b"], s["m"], s["k"], s["n"]], "ms": [round(x, 5) for x in ms],
            "median_ms": round(med, 5), "min_ms": round(min(ms), 5),
            "spread_pct": round(100 * (max(ms) - min(ms)) / med, 3),
            "us_per_call": round(med * 1e3, 2),
            "tflops": round(f / 1e12 / (med / 1e3), 3),
            "gbs": round(min_bytes(s["b"], s["m"], s["k"], s["n"]) / 1e9 / (med / 1e3), 2),
            "fold_s_at_this_rate": round(s["calls"] * med / 1e3, 5),
            "insitu_s": s["insitu_s"]}
    res["env"]["t_close"] = time.time()
    res["env"]["clk_end"] = clk_end
    out = HERE / ("ditrate_%s.json" % a.tag)
    out.write_text(json.dumps(res, indent=1))

    print("\n%-18s %9s %9s %9s %8s %10s %10s"
          % ("arm", "us/call", "TFLOP/s", "GB/s", "spread%", "fold_s", "insitu_s"))
    for kk, v in res["arms"].items():
        print("%-18s %9.2f %9.2f %9.2f %8.2f %10.4f %10.4f"
              % (kk, v["us_per_call"], v["tflops"], v["gbs"], v["spread_pct"],
                 v["fold_s_at_this_rate"], v["insitu_s"]))
    print("\nA/A twins")
    for nm in SHAPES:
        for pair in (("ship", "ship_aa"), ("both_l1", "both_l1_aa")):
            x, y = res["arms"]["%s/%s" % (nm, pair[0])], res["arms"]["%s/%s" % (nm, pair[1])]
            d = 100 * abs(x["median_ms"] - y["median_ms"]) / x["median_ms"]
            print("  %-12s %-10s %.3f %%  %s" % (nm, pair[0], d, "OK" if d <= 3 else "REFUSE"))
    print("\nwrote", out, "clock at close", clk_end, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
