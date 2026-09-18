#!/usr/bin/env python3
"""TRIX floor/roof row: re-measure the roofs this campaign scores against, on THIS card,
in the kernel config the shipped trimul actually runs.

Why every roof in this campaign disagrees with the next one is the question. Three passes on
qb2 card 0 have published a square compute roof of 136-137, 109.56 and 121.95 TFLOP/s, and a
combined DRAM roof of 393.5, 363.2/375 and 442.9 GB/s. A number without a DURING-sampled clock
is not a measurement on Blackhole, and none of those three recorded one.

So: the clock is FORCED by this process (never a helper -- a helper that is signalled away ends
the force before the measurement does) and read from sysfs on a thread throughout. Every arm
carries an A/A twin and the session refuses to publish an arm whose twin spreads more than 3 %.
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
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

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
    except OSError:
        return None


class ClockTrace:
    """Samples the granted chip's AICLK on a thread for the whole session."""

    def __init__(self, node, period=0.25):
        self.node, self.period, self.samples = node, period, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            c = read_aiclk(self.node)
            if c is not None:
                self.samples.append((time.time(), c))
            time.sleep(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def window(self, t0, t1):
        xs = [c for t, c in self.samples if t0 <= t <= t1]
        if not xs:
            return None
        s = sorted(xs)
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1], "n": len(s)}

    def summary(self):
        return self.window(0, time.time())


import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
CLK: ClockTrace | None = None
RESULTS: dict = {}


def ckc(fp32, pl1, fid=ttnn.MathFidelity.HiFi4):
    dev = get_device()
    kcls = (ttnn.types.WormholeComputeKernelConfig
            if dev.arch() == ttnn.Arch.WORMHOLE_B0 else ttnn.types.BlackholeComputeKernelConfig)
    return kcls(math_fidelity=fid, math_approx_mode=False,
                fp32_dest_acc_en=fp32, packer_l1_acc=pl1)


def pipe(fn, dev, warm=3, inflight=6):
    """Pipelined, no per-call sync -- the methodology the two prior roof passes used."""
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    best, t_start = None, time.time()
    for _ in range(3):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(inflight)]
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / inflight
        best = ms if best is None else min(best, ms)
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    return best, (t_start, time.time())


def serial(fn, dev, warm=3, reps=7):
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts, t_start = [], time.time()
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if r is not None:
            ttnn.deallocate(r)
    return st.median(ts), min(ts), (t_start, time.time())


def record(tag, fn, dev, *, flop=None, gbytes=None, inflight=6, twin=True):
    """One arm plus its A/A twin, both through the same descriptor."""
    p1, w1 = pipe(fn, dev, inflight=inflight)
    s1, s1min, ws1 = serial(fn, dev)
    row = {"pipe_ms": p1, "serial_med_ms": s1, "serial_min_ms": s1min,
           "clock": CLK.window(*w1), "clock_serial": CLK.window(*ws1)}
    if twin:
        p2, _ = pipe(fn, dev, inflight=inflight)
        row["pipe_ms_aa"] = p2
        row["aa_spread_pct"] = 100.0 * abs(p2 - p1) / min(p1, p2)
    if flop:
        row["tflops_pipe"] = flop / 1e12 / (p1 / 1e3)
        row["tflops_serial"] = flop / 1e12 / (s1 / 1e3)
    if gbytes:
        row["gbs_pipe"] = gbytes / (p1 / 1e3)
        row["gbs_serial"] = gbytes / (s1 / 1e3)
    RESULTS[tag] = row
    aa = f" aa {row.get('aa_spread_pct', float('nan')):.2f}%"
    rate = ""
    if flop:
        rate = f"  {row['tflops_pipe']:.2f} TF/s pipe / {row['tflops_serial']:.2f} serial"
    if gbytes:
        rate = f"  {row['gbs_pipe']:.1f} GB/s pipe / {row['gbs_serial']:.1f} serial"
    c = row["clock"] or {}
    print(f"  {tag:52s} {p1:8.4f} ms pipe {s1:8.4f} serial{rate}{aa} "
          f"clk {c.get('min')}/{c.get('median')}/{c.get('max')} n={c.get('n')}", flush=True)
    return row


def main() -> int:
    global CLK
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "roofs_qb2c0.json")
    a = ap.parse_args()

    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    before = read_aiclk(a.node)
    status = "0x%02X" % _smc(fd, FORCE_AICLK, a.clock)
    print(f"node{a.node} FORCE_AICLK({a.clock}) status={status} before={before}", flush=True)

    def _release():
        try:
            _smc(fd, FORCE_AICLK, 0)
            print(f"node{a.node} clock force released", flush=True)
        except Exception as e:                                                # noqa: BLE001
            print(f"node{a.node} RELEASE FAILED {e!r}", flush=True)

    atexit.register(_release)
    t_ramp = time.time()
    while read_aiclk(a.node) != a.clock and time.time() - t_ramp < 10.0:
        time.sleep(0.01)
    now = read_aiclk(a.node)
    if now != a.clock:
        print(f"REFUSING: node{a.node} at {now} MHz, not {a.clock}", flush=True)
        return 3
    print(f"node{a.node} pinned at {now} MHz after {1000*(time.time()-t_ramp):.0f} ms", flush=True)

    CLK = ClockTrace(a.node).start()
    dev = get_device()
    print(f"grid {COMPUTE_GRID_MAIN}  arch {dev.arch()}", flush=True)
    RESULTS["_meta"] = {"node": a.node, "forced_mhz": a.clock, "grid": list(COMPUTE_GRID_MAIN),
                        "ttnn": getattr(ttnn, "__version__", "?"),
                        "loadavg": os.getloadavg(), "t0": time.time()}

    SHIPPED = ckc(True, True)

    # ---- 1. known-answer control -------------------------------------------------------
    # 2048^3 = 17,179,869,184 FLOP. Two prior sessions on this card measured 0.1520-0.1537 ms.
    print("\n=== 1. known-answer control: 2048^3, shipped ckc, DRAM out ===", flush=True)
    k = 2048
    a2 = ttnn.from_torch(torch.zeros(k, k), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    record("control/2048cube_shipped_dram", lambda: ttnn.matmul(
        a2, a2, compute_kernel_config=SHIPPED, memory_config=DRAM,
        core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16), dev, flop=2.0 * k ** 3, inflight=4)
    ttnn.deallocate(a2)

    # ---- 2. square compute roof, kernel-config matrix -----------------------------------
    print("\n=== 2. square compute roof: the same cube across kernel configs ===", flush=True)
    for k in (2048, 4096):
        src = ttnn.from_torch(torch.zeros(k, k), layout=ttnn.TILE_LAYOUT, device=dev,
                              dtype=ttnn.bfloat16, memory_config=DRAM)
        for name, fp32, pl1 in (("SHIPPED_fp32T_pl1T", True, True),
                                ("fp32F_pl1T", False, True),
                                ("fp32T_pl1F", True, False),
                                ("fp32F_pl1F", False, False)):
            cfg = ckc(fp32, pl1)
            record(f"compute/{k}cube/{name}", lambda c=cfg, s=src: ttnn.matmul(
                s, s, compute_kernel_config=c, memory_config=DRAM,
                core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16), dev,
                flop=2.0 * k ** 3, inflight=4)
        ttnn.deallocate(src)

    # ---- 3. DRAM roofs -------------------------------------------------------------------
    print("\n=== 3. DRAM roofs: clone ladder, read, write ===", flush=True)
    for mib_ in (24, 48, 128, 256):
        rows = int(mib_ * 2 ** 20 / 2 / 1024)
        s = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        record(f"dram/combined_clone_{mib_}MiB", lambda s=s: ttnn.clone(s, memory_config=DRAM),
               dev, gbytes=2 * mib_ / 1024, inflight=4)
        ttnn.deallocate(s)
    for mib_ in (24, 48):
        rows = int(mib_ * 2 ** 20 / 2 / 1024)
        s = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        record(f"dram/read_DRAM_to_L1_{mib_}MiB", lambda s=s: ttnn.clone(s, memory_config=L1),
               dev, gbytes=mib_ / 1024, inflight=max(1, int(100 // mib_)))
        ttnn.deallocate(s)
        s = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        record(f"dram/write_L1_to_DRAM_{mib_}MiB", lambda s=s: ttnn.clone(s, memory_config=DRAM),
               dev, gbytes=mib_ / 1024, inflight=4)
        ttnn.deallocate(s)
    # second instrument for the combined roof: a 3-operand elementwise add, 2 reads + 1 write,
    # arithmetic intensity 1/6 FLOP per byte, so it cannot be anything but bandwidth.
    for mib_ in (64, 128):
        rows = int(mib_ * 2 ** 20 / 2 / 1024)
        x = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        y = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        record(f"dram/add3_{mib_}MiB", lambda x=x, y=y: ttnn.add(x, y, memory_config=DRAM),
               dev, gbytes=3 * mib_ / 1024, inflight=4)
        ttnn.deallocate(x)
        ttnn.deallocate(y)

    # ---- 4. the three trimul matmul class rates, production configs ----------------------
    print("\n=== 4. trimul matmul class rates, production call sites ===", flush=True)
    N, C_Z, HID = 512, 256, 256
    z = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    for G, nout in ((1, 4 * 32), (8, 4 * 256)):
        w = ttnn.from_torch(torch.randn(C_Z, nout), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        record(f"class/in_proj_G{G}_[1,512,512,256]@[256,{nout}]",
               lambda w=w: T._in_proj_matmul(z, w, SHIPPED, DRAM), dev,
               flop=2.0 * N * N * C_Z * nout, inflight=3)
        ttnn.deallocate(w)
    # the contraction, one channel block of 32 and the full 256, production program config
    pc = T._triangle_mul_program_config((N + 31) // 32)
    for C in (32, 256):
        aa = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        bb = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        record(f"class/contraction_C{C}_[1,{C},512,512]sq",
               lambda aa=aa, bb=bb: ttnn.matmul(
                   aa, bb, compute_kernel_config=SHIPPED, memory_config=DRAM,
                   program_config=pc, dtype=ttnn.bfloat16), dev,
               flop=2.0 * C * N * N * N, inflight=2 if C == 32 else 1)
        ttnn.deallocate(aa)
        ttnn.deallocate(bb)
    # the output projection through the tuned 1D-mcast config production uses
    xh = ttnn.from_torch(torch.randn(1, N, N, HID), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    wo = ttnn.from_torch(torch.randn(HID, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    record("class/out_proj_[1,512,512,256]@[256,256]",
           lambda: T._pair_proj_linear(xh, wo, SHIPPED, ttnn.bfloat16), dev,
           flop=2.0 * N * N * HID * C_Z, inflight=3)
    ttnn.deallocate(xh)
    ttnn.deallocate(wo)
    ttnn.deallocate(z)

    RESULTS["_meta"]["clock_session"] = CLK.summary()
    RESULTS["_meta"]["loadavg_end"] = os.getloadavg()
    RESULTS["_meta"]["t1"] = time.time()
    CLK.stop()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nwrote {a.out}", flush=True)
    print("SESSION CLOCK:", RESULTS["_meta"]["clock_session"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
