#!/usr/bin/env python3
"""Why three passes on ONE card published three different roofs: the clock.

`trimul-bottleneck-rootcause` measured the square compute roof at 109.56 TFLOP/s,
`trimul-absolute-optimal` at 121.95 on the same card 0, and the WARROOM carried 136-137.
None of the three recorded the AICLK it measured at. This walks the identical probes across
a forced clock ladder, so the roof, the three trimul matmul class rates and the whole trimul
module are all read at 800 / ~1063 / 1350 MHz in one process on one card.

Every timed region is bracketed by a sysfs AICLK read on the measuring thread, so no number
here depends on a sampler thread winning the GIL.
"""
from __future__ import annotations

import argparse, atexit, fcntl, json, os, statistics as st, struct, sys, threading, time
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


import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
NODE = 0
ROWS: list = []


def serial(fn, dev, warm=3, reps=7):
    """Median of `reps` synced calls, with the clock read on THIS thread each rep."""
    for _ in range(warm):
        r = fn()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts, clks = [], []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        c0 = read_aiclk(NODE)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        clks.append((c0, read_aiclk(NODE)))
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    flat = [c for pair in clks for c in pair if c is not None]
    return st.median(ts), min(ts), max(ts), {"min": min(flat), "max": max(flat),
                                             "median": st.median(flat), "n": len(flat)}


def arm(clk_req, tag, fn, dev, *, flop=None, gbytes=None, reps=7):
    med, lo, hi, clk = serial(fn, dev, reps=reps)
    row = {"clock_req": clk_req, "tag": tag, "ms_med": med, "ms_min": lo, "ms_max": hi,
           "spread_pct": 100.0 * (hi - lo) / lo, "clk": clk}
    if flop:
        row["tflops"] = flop / 1e12 / (med / 1e3)
    if gbytes:
        row["gbs"] = gbytes / (med / 1e3)
    ROWS.append(row)
    rate = ""
    if flop:
        rate = f" {row['tflops']:8.2f} TF/s"
    if gbytes:
        rate = f" {row['gbs']:8.1f} GB/s"
    print(f"  [{clk['median']:>4} MHz {clk['min']}-{clk['max']}] {tag:44s} "
          f"{med:9.4f} ms{rate}  spread {row['spread_pct']:.2f}%", flush=True)
    return row


def main() -> int:
    global NODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--clocks", default="800,1050,1350")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c-z", dest="c_z", type=int, default=256)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "clkladder_qb2c0.json")
    a = ap.parse_args()
    NODE = a.node

    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)

    def set_clock(mhz):
        _smc(fd, FORCE_AICLK, mhz)
        t0 = time.time()
        while time.time() - t0 < 10.0:
            c = read_aiclk(a.node)
            if c is not None and abs(c - mhz) <= 2:
                return c
            time.sleep(0.01)
        return read_aiclk(a.node)

    atexit.register(lambda: _smc(fd, FORCE_AICLK, 0))

    dev = get_device()
    N = a.n
    ckc = T.trunk_compute_kernel_config(
        ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True))
    # The module straight from its own weight contract, so nothing outside this file decides
    # the shape. `reference.TriangleMultiplicationOutgoing` fixes hidden == c_z: p_in and g_in
    # are dim -> 2*dim and the chunk halves that, so a and b are dim wide.
    c_z = a.c_z
    torch.manual_seed(0)
    sd = {
        "norm_in.weight": torch.ones(c_z), "norm_in.bias": torch.zeros(c_z),
        "norm_out.weight": torch.ones(c_z), "norm_out.bias": torch.zeros(c_z),
        "p_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
        "g_in.weight": torch.randn(2 * c_z, c_z) * 0.03,
        "p_out.weight": torch.randn(c_z, c_z) * 0.03,
        "g_out.weight": torch.randn(c_z, c_z) * 0.03,
    }
    tm = T.TriangleMultiplication(ending=False, state_dict=sd, compute_kernel_config=ckc)
    hid = tm._hidden
    print(f"grid {COMPUTE_GRID_MAIN}  N={N} c_z={c_z} hidden={hid}", flush=True)
    torch.manual_seed(0)

    meta = {"node": a.node, "grid": list(COMPUTE_GRID_MAIN), "n": N, "c_z": c_z, "hidden": hid,
            "loadavg_start": os.getloadavg()}

    for req in [int(x) for x in a.clocks.split(",")]:
        got = set_clock(req)
        print(f"\n=== FORCE_AICLK({req}) -> sysfs reads {got} MHz ===", flush=True)
        # 1. square compute roof, the shipped kernel config
        for k in (2048, 4096):
            s = ttnn.from_torch(torch.zeros(k, k), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            arm(req, f"compute/{k}cube_shipped", lambda s=s: ttnn.matmul(
                s, s, compute_kernel_config=ckc, memory_config=DRAM,
                core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16), dev, flop=2.0 * k ** 3)
            ttnn.deallocate(s)
        # 2. DRAM combined roof, two instruments
        rows = int(128 * 2 ** 20 / 2 / 1024)
        s = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, "dram/clone_128MiB", lambda s=s: ttnn.clone(s, memory_config=DRAM), dev,
            gbytes=2 * 128 / 1024)
        s2 = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, "dram/add3_128MiB", lambda s=s, s2=s2: ttnn.add(s, s2, memory_config=DRAM), dev,
            gbytes=3 * 128 / 1024)
        ttnn.deallocate(s)
        ttnn.deallocate(s2)
        # 3. the three trimul matmul class rates at production call sites
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w8 = ttnn.from_torch(torch.randn(c_z, 4 * hid), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, f"class/in_proj_G8[{c_z}->{4*hid}]",
            lambda: T._in_proj_matmul(z, w8, ckc, DRAM), dev,
            flop=2.0 * N * N * c_z * 4 * hid, reps=5)
        ttnn.deallocate(w8)
        pc = T._triangle_mul_program_config((N + 31) // 32)
        aa = ttnn.from_torch(torch.randn(1, hid, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        bb = ttnn.from_torch(torch.randn(1, hid, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, f"class/contraction_C{hid}", lambda: ttnn.matmul(
            aa, bb, compute_kernel_config=ckc, memory_config=DRAM,
            program_config=pc, dtype=ttnn.bfloat16), dev, flop=2.0 * hid * N * N * N, reps=5)
        ttnn.deallocate(aa)
        ttnn.deallocate(bb)
        xh = ttnn.from_torch(torch.randn(1, N, N, hid), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        wo = ttnn.from_torch(torch.randn(hid, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, f"class/out_proj[{hid}->{c_z}]",
            lambda: T._pair_proj_linear(xh, wo, ckc, ttnn.bfloat16), dev,
            flop=2.0 * N * N * hid * c_z, reps=5)
        ttnn.deallocate(xh)
        ttnn.deallocate(wo)
        # 4. the production module itself, standalone, at this clock
        arm(req, f"module/trimul_start_{N}aa_nomask", lambda: tm(z), dev, reps=5)
        msk = ttnn.from_torch(torch.ones(1, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                              dtype=ttnn.bfloat16, memory_config=DRAM)
        arm(req, f"module/trimul_start_{N}aa_masked", lambda: tm(z, msk), dev, reps=5)
        ttnn.deallocate(msk)
        ttnn.deallocate(z)

    meta["loadavg_end"] = os.getloadavg()
    set_clock(0)
    a.out.write_text(json.dumps({"meta": meta, "rows": ROWS}, indent=1, default=str))
    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
