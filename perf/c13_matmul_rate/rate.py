#!/usr/bin/env python3
"""Where the 82.5 % goes on the fold's thin-K matmul shapes, measured by ablation.

The question this row owns: the fold's matmul class runs at 21.38 TFLOP/s against a same-session
122.29 TFLOP/s dense cube, and 10.0 s needs 35.49. c10-fold-census already showed the class rate is
a spread, not a level -- the 768-inner feedforward shapes reach 41-66 TFLOP/s while the two keys
that carry the thin-K mass sit at 16.49 and 12.15. So the 82.5 % has to be decomposed on those two
keys, not on the class average.

The instrument is an ABLATION, not a counter. Four campaigns here were derailed by counters that
returned confident wrong numbers, and the one time this project root-caused a matmul that saturated
neither roof (`tt-bio-qkv-op-utilization-rootcause`, 27 % of compute and 33 % of bandwidth at once)
it was done by removing the DRAM destination and measuring the delta -- no device profiler
involved. Each arm below removes exactly one term and the delta prices it:

  out_l1     output stays in L1               -> prices the DRAM WRITE term
  in_l1      operands start in L1             -> prices the DRAM READ term
  both_l1    neither operand nor result in DRAM -> prices the whole DRAM term
  fid_hifi2  half the FPU passes per tile     -> prices the MATH term (bf16 needs HiFi2, so this
             fid_lofi   a quarter of them         is also the highest-accuracy arm that is faster)
  flat2d     leading dims folded into M       -> prices the batch loop
  nogrid     no core_grid passed              -> prices our config against ttnn's default routing
  k*         K scaled at fixed M, N, batch    -> separates per-call fixed cost from streaming rate

An arm that does not move is as informative as one that does: if LoFi buys nothing the FPU is not
issuing at the limit, and if out_l1 buys most of the gap the term is the writeback.

KNOWN-ANSWER CONTROLS, in the same session, before any arm is believed (the gate this row is held
to): a 2048^3 cube whose FLOP count is exactly 17,179,869,184 and which two prior sessions on THIS
card measured at 0.1520-0.1537 ms; an 8192^3 cube at exactly 1,099,511,627,776 FLOPs; a starved
8192^2 add at exactly 402,653,184 bytes against the campaign's 442.9 GB/s DRAM roof. Bytes are
counted from the shapes, and the shapes are dense and each operand is touched once, so the byte
count is a closed form rather than a counter reading -- the failure mode the brief names (bytes
deduped on tensor id instead of buffer address) cannot occur because no counter is read at all.
Every arm carries an A/A twin and the session is refused if a twin spreads more than 3 %.

Clock is forced and sampled by perf/c12_genop_rate/clk.py in a separate process across the session.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import statistics as st
import sys
import time
import fcntl
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ARC FORCE_AICLK over tt-kmd's SMC message queue, lifted from perf/c12_genop_rate/clk.py. The
# force is held by THIS process, not by a helper: on 2026-09-18 the helper that both forced and
# sampled was signalled away the instant this process opened the chip, the force was released
# 21 s into a 4-minute session, and the clock record still read "99.8 % at 1350 MHz" because the
# samples stopped at the same moment. A force that lives in the measuring process cannot end
# before the measurement does.
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

HERE = Path(__file__).resolve().parent
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# The two keys that carry the fold's thin-K matmul mass, with their executed call counts from
# perf/roof_launch/op_census_512.json (the 465,664-call census c10-fold-census weighted its budget
# by). Key A appears under two census strings that differ only in operand print order
# (8,448 + 9,472); B under one.
KEYS = {
    "A": dict(batch=16, m=512, k=128, n=512, calls=8448 + 9472, kmult=(2, 4, 8, 16),
              census_tflops=16.49,
              census="ttnn.linear|out=1x16x512x512|in=1x16x512x128,128x512"),
    "B": dict(batch=16, m=512, k=512, n=128, calls=8448, kmult=(2, 4, 8),
              census_tflops=12.15,
              census="ttnn.linear|out=1x16x512x128|in=1x16x512x512,512x128"),
}


def flops(batch, m, k, n):
    return 2.0 * batch * m * k * n


def min_bytes(batch, m, k, n, elem=2):
    """Compulsory DRAM traffic: in0 once, the shared weight once, the result once."""
    return float(elem) * (batch * m * k + k * n + batch * m * n)


def tile_flops(batch, m, k, n):
    """The same FLOP count off the tile grid, as the second of two independent counts."""
    mt, kt, nt = m // 32, k // 32, n // 32
    assert m % 32 == 0 and k % 32 == 0 and n % 32 == 0
    return float(batch) * mt * kt * nt * 2 * 32 * 32 * 32


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
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3,
                    help="physical /dev/tenstorrent/N whose clock this process forces")
    ap.add_argument("--out", type=Path, default=HERE / "rate_qb2c3.json")
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T

    clk_fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    clk_before = read_aiclk(a.node)
    clk_status = "0x%02X" % _smc(clk_fd, FORCE_AICLK, a.clock)
    print("node%d FORCE_AICLK(%d) status=%s before=%s after=%s"
          % (a.node, a.clock, clk_status, clk_before, read_aiclk(a.node)), flush=True)

    def _release():
        try:
            _smc(clk_fd, FORCE_AICLK, 0)
            print("node%d clock force released" % a.node, flush=True)
        except Exception as e:                                                 # noqa: BLE001
            print("node%d RELEASE FAILED %r" % (a.node, e), flush=True)

    atexit.register(_release)

    t_ramp = time.time()
    while read_aiclk(a.node) != a.clock and time.time() - t_ramp < 10.0:
        time.sleep(0.01)
    ramp_ms = 1000.0 * (time.time() - t_ramp)
    now = read_aiclk(a.node)
    if now != a.clock:
        print("REFUSING to measure: node%d is at %s MHz, not the requested %d, %.0f ms after "
              "FORCE_AICLK returned %s" % (a.node, now, a.clock, ramp_ms, clk_status), flush=True)
        return 2
    print("node%d reached %d MHz after %.0f ms; measurement window opens now"
          % (a.node, a.clock, ramp_ms), flush=True)
    t_measure_start = time.time()

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    cores = g.x * g.y
    grid = T.CORE_GRID_MAIN            # module attribute AFTER device open, never the import-time name
    print("grid %dx%d = %d cores, CORE_GRID_MAIN=%r" % (g.x, g.y, cores, grid), flush=True)

    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid, fp32=True, pl1=True):
        return kcls(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32,
                    packer_l1_acc=pl1)

    # The fold's own config for this class: HiFi4 / fp32_dest_acc_en=1 / packer_l1_acc=1
    # (c12-matmul-key-attribution, read off the executed graph).
    KC_SHIP = kc(ttnn.MathFidelity.HiFi4)
    KC_HIFI2 = kc(ttnn.MathFidelity.HiFi2)
    KC_LOFI = kc(ttnn.MathFidelity.LoFi)
    # the config the campaign's published roofs use (scripts/profiling/roofline_bh.py:32)
    KC_ROOF = kc(ttnn.MathFidelity.HiFi4, fp32=False, pl1=False)

    torch.manual_seed(0)

    def dev(t, mc=DRAM):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=mc)

    arms = []      # (name, callable, kind, amount, group)

    # ---- session controls, known answer -------------------------------------------------
    c2 = dev(torch.randn(2048, 2048) * 0.05)
    c2b = dev(torch.randn(2048, 2048) * 0.05)
    c8 = dev(torch.randn(8192, 8192) * 0.05)
    c8b = dev(torch.randn(8192, 8192) * 0.05)
    add_a = dev(torch.randn(8192, 8192) * 0.05)
    add_b = dev(torch.randn(8192, 8192) * 0.05)

    F2048, F8192 = flops(1, 2048, 2048, 2048), flops(1, 8192, 8192, 8192)
    assert F2048 == 17179869184.0, F2048
    assert F8192 == 1099511627776.0, F8192
    assert tile_flops(1, 2048, 2048, 2048) == F2048
    assert tile_flops(1, 8192, 8192, 8192) == F8192
    BW_ADD = 3 * 8192 * 8192 * 2
    assert BW_ADD == 402653184, BW_ADD

    for nm in ("ctl_cube2048", "ctl_cube2048_aa"):
        arms.append((nm, lambda: ttnn.matmul(c2, c2b, compute_kernel_config=KC_ROOF,
                                             memory_config=DRAM), "flops", F2048, "ctl"))
    for nm in ("ctl_cube8192", "ctl_cube8192_aa"):
        arms.append((nm, lambda: ttnn.matmul(c8, c8b, compute_kernel_config=KC_ROOF,
                                             memory_config=DRAM), "flops", F8192, "ctl"))
    # The cube under the MODEL'S OWN kernel config, which is the config every key arm below runs
    # and therefore the only legitimate denominator for "x % of the dense cube". c10-fold-census's
    # 122.29 TFLOP/s was taken with af2.compute_kernel_config() (HiFi4, fp32_dest_acc_en=True,
    # packer_l1_acc=True) on node 1; the roofline script's config (fp32 acc off, packer L1 acc off)
    # is a different roof, and mixing the two is how a ratio gets quoted against the wrong roof.
    for nm in ("ctl_cube8192_ship", "ctl_cube8192_ship_aa"):
        arms.append((nm, lambda: ttnn.matmul(c8, c8b, compute_kernel_config=KC_SHIP,
                                             memory_config=DRAM), "flops", F8192, "ctl"))
    for nm in ("ctl_cube2048_ship", "ctl_cube2048_ship_aa"):
        arms.append((nm, lambda: ttnn.matmul(c2, c2b, compute_kernel_config=KC_SHIP,
                                             memory_config=DRAM), "flops", F2048, "ctl"))
    # and the two fidelities below it, so the session knows whether its own cube is FPU-limited
    for nm, kcf in (("ctl_cube8192_hifi2", KC_HIFI2), ("ctl_cube8192_lofi", KC_LOFI)):
        arms.append((nm, (lambda kk: lambda: ttnn.matmul(c8, c8b, compute_kernel_config=kk,
                                                         memory_config=DRAM))(kcf),
                     "flops", F8192, "ctl"))
    for nm in ("ctl_bwadd", "ctl_bwadd_aa"):
        arms.append((nm, lambda: ttnn.add(add_a, add_b, memory_config=DRAM),
                     "bytes", float(BW_ADD), "ctl"))

    # ---- the two keys, with their ablation ladders --------------------------------------
    holders = []
    for kn, spec in KEYS.items():
        b, m, k, n = spec["batch"], spec["m"], spec["k"], spec["n"]
        F = flops(b, m, k, n)
        assert F == tile_flops(b, m, k, n), (kn, F)
        B = min_bytes(b, m, k, n)
        spec["flops"], spec["min_bytes"] = F, B

        x_d = dev(torch.randn(1, b, m, k) * 0.05)
        w_d = dev(torch.randn(k, n) * 0.05)
        x_l = dev(torch.randn(1, b, m, k) * 0.05, mc=L1)
        w_l = dev(torch.randn(k, n) * 0.05, mc=L1)
        x_f = dev(torch.randn(b * m, k) * 0.05)
        holders += [x_d, w_d, x_l, w_l, x_f]

        def mk(xa, wa, mc, kcfg, use_grid=True):
            def f():
                kw = dict(compute_kernel_config=kcfg, memory_config=mc)
                if use_grid:
                    kw["core_grid"] = grid
                return ttnn.linear(xa, wa, **kw)
            return f

        g_ = "key" + kn
        arms.append(("%s_ship" % kn, mk(x_d, w_d, DRAM, KC_SHIP), "flops", F, g_))
        arms.append(("%s_ship_aa" % kn, mk(x_d, w_d, DRAM, KC_SHIP), "flops", F, g_))
        arms.append(("%s_out_l1" % kn, mk(x_d, w_d, L1, KC_SHIP), "flops", F, g_))
        arms.append(("%s_in_l1" % kn, mk(x_l, w_l, DRAM, KC_SHIP), "flops", F, g_))
        arms.append(("%s_both_l1" % kn, mk(x_l, w_l, L1, KC_SHIP), "flops", F, g_))
        arms.append(("%s_both_l1_aa" % kn, mk(x_l, w_l, L1, KC_SHIP), "flops", F, g_))
        arms.append(("%s_fid_hifi2" % kn, mk(x_d, w_d, DRAM, KC_HIFI2), "flops", F, g_))
        arms.append(("%s_fid_lofi" % kn, mk(x_d, w_d, DRAM, KC_LOFI), "flops", F, g_))
        arms.append(("%s_nofp32" % kn, mk(x_d, w_d, DRAM,
                                          kc(ttnn.MathFidelity.HiFi4, fp32=False)),
                     "flops", F, g_))
        arms.append(("%s_nogrid" % kn, mk(x_d, w_d, DRAM, KC_SHIP, use_grid=False),
                     "flops", F, g_))
        arms.append(("%s_flat2d" % kn, mk(x_f, w_d, DRAM, KC_SHIP), "flops", F, g_))
        arms.append(("%s_flat2d_aa" % kn, mk(x_f, w_d, DRAM, KC_SHIP), "flops", F, g_))
        # flat AND deep: the batch dim gone and K taken to the dense regime, i.e. the same
        # M,N with nothing thin about it. This is the ceiling these M,N can reach at all.

        for gx, gy in ((11, 10), (8, 8), (6, 6), (4, 4), (2, 2), (1, 1)):
            cg = ttnn.CoreGrid(x=gx, y=gy)

            def mkg(xa, wa, mc, cgg):
                def f():
                    return ttnn.linear(xa, wa, compute_kernel_config=KC_SHIP,
                                       memory_config=mc, core_grid=cgg)
                return f
            arms.append(("%s_g%dx%d" % (kn, gx, gy), mkg(x_d, w_d, DRAM, cg), "flops", F, g_))
            arms.append(("%s_g%dx%d_l1" % (kn, gx, gy), mkg(x_l, w_l, L1, cg), "flops", F, g_))

        # K ladder: fixed batch, M, N; K scaled. Separates a per-call fixed cost from the rate.
        for mult in spec["kmult"]:
            kk = k * mult
            xk = dev(torch.randn(1, b, m, kk) * 0.05)
            xkf = dev(torch.randn(b * m, kk) * 0.05)
            wk = dev(torch.randn(kk, n) * 0.05)
            holders += [xk, xkf, wk]
            Fk = flops(b, m, kk, n)
            arms.append(("%s_k%dx" % (kn, mult), mk(xk, wk, DRAM, KC_SHIP), "flops", Fk, g_))
            arms.append(("%s_k%dx_flat" % (kn, mult), mk(xkf, wk, DRAM, KC_SHIP),
                         "flops", Fk, g_))

    # ---- run, interleaved rep by rep ---------------------------------------------------
    print("arms=%d  reps=%d warm=%d" % (len(arms), a.reps, a.warm), flush=True)
    # Warm every arm first, and drop an arm the part refuses (an L1 arm can miss the fit). A
    # dropped arm is reported as such, never silently absent.
    refused = {}
    live = []
    for nm, fn, kind, amt, grp in arms:
        try:
            timed(fn, device, 0, a.warm)
            live.append((nm, fn, kind, amt, grp))
        except Exception as e:                                             # noqa: BLE001
            refused[nm] = repr(e)[:300]
            print("  REFUSED %s: %s" % (nm, repr(e)[:160]), flush=True)
    arms = live
    res = {nm: [] for nm, *_ in arms}
    for rep in range(a.reps):
        for nm, fn, kind, amt, grp in arms:
            res[nm] += timed(fn, device, 1, 0)
        print("  rep %d/%d done" % (rep + 1, a.reps), flush=True)

    t_measure_end = time.time()
    out = {"host": os.uname().nodename, "cores": cores, "clock_mhz": a.clock,
           "node": a.node, "clock_force": {"status": clk_status, "before": clk_before,
                                            "during_end": read_aiclk(a.node)},
           "t_measure_start": t_measure_start, "t_measure_end": t_measure_end,
           "grid": repr(grid), "reps": a.reps, "warm": a.warm,
           "controls": {"F2048": F2048, "F8192": F8192, "BW_ADD": BW_ADD},
           "keys": {k: {kk: vv for kk, vv in v.items()} for k, v in KEYS.items()},
           "refused": refused, "arms": {}}
    meta = {nm: (kind, amt, grp) for nm, fn, kind, amt, grp in arms}
    for nm, ms in res.items():
        kind, amt, grp = meta[nm]
        lo, med = min(ms), st.median(ms)
        row = {"kind": kind, "amount": amt, "group": grp, "ms_min": lo, "ms_med": med,
               "ms_all": ms, "spread_pct": 100.0 * (max(ms) - lo) / lo,
               "Mcycles": lo * 1e-3 * a.clock * 1e6 / 1e6}
        row["tflops"] = amt / (lo * 1e-3) / 1e12 if kind == "flops" else None
        row["gbs"] = amt / (lo * 1e-3) / 1e9 if kind == "bytes" else None
        out["arms"][nm] = row

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print("\n%-16s %10s %10s %10s %10s" % ("arm", "ms_min", "ms_med", "TFLOP/s", "GB/s"))
    for nm, fn, kind, amt, grp in arms:
        r = out["arms"][nm]
        print("%-16s %10.4f %10.4f %10s %10s"
              % (nm, r["ms_min"], r["ms_med"],
                 "%.2f" % r["tflops"] if r["tflops"] else "-",
                 "%.1f" % r["gbs"] if r["gbs"] else "-"))
    print("\nwrote %s" % a.out)
    print("measurement span %.1f s" % (t_measure_end - t_measure_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
