#!/usr/bin/env python3
"""Perf screen of the four RF3 shapes `_short_m_proj_config` SERVES but was never fitted on.

Pass 4 counted the firing: a 117 aa RF3 fold serves 4,419 of 4,419 calls on
`(1,117,768)x(768,1536)`, `(1,117,1536)x(1536,768)`, `(1,29,32,128)x(128,256)` and
`(1,29,32,256)x(256,128)` (`fire_rf3_shim.json`). None of the four is in the fitted set, which was
Boltz-2's 768 -> 768/1536/3072 and 1536 -> 768 DiT projections. Accuracy cleared on both models,
but pass 3's own lesson 9 says accuracy is the wrong instrument for this question: the rule was
caught regressing `1,512,768,3072` by 1.114x only by re-measuring the shapes it CHANGES, and a
1.1x perf regression is invisible in an lDDT reading.

So this is the same screen, on RF3's groups. The arms are the two routings the shipped call site
actually takes, not two hand-written configs:

    base   ttnn.linear(x, w, core_grid=CORE_GRID_MAIN)          `_proj` when the rule declines
    bw     ttnn.linear(x, w, program_config=_short_m_proj_config(x, w))   when it serves
    aa     byte-identical twin of base, interleaved with the other two

`_short_m_proj_config` is CALLED, with the module flag flipped through its own setter, so the arm
is the shipped decision and not my reading of it. A rule change that made the function decline
would show here as an A/A, which is the correct answer for a screen.

Every timed region issues CHAIN calls between syncs. Session s1 measured 25.0 us of host round trip
against a 20.66 us kernel, so a one-op-per-sync bench cannot state a rate; chaining divides it away.

KNOWN-ANSWER CONTROLS, before any arm is believed: the 8192^3 cube under the arms' own kernel
config with its FLOP count asserted equal by two independent routes, the starved 8192^2 add of
exactly 402,653,184 bytes, and the A/A twin per shape. The clock is forced to 1350 over the ARC
queue and sampled at 1 kHz for the whole session by a reader holding no device fd.
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
HERE = Path(__file__).resolve().parent
_IOC = (0xFA << 8) | 17
_POST, _POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33

# (x shape, w shape, calls per 117 aa fold) from fire_rf3_shim.json. The call counts are carried
# so the table can rank the groups; they are NOT turned into seconds, because this bench has no
# RF3 in-fold us/call to use as a base and a ratio is not seconds.
RF3_GROUPS = [
    ((1, 117, 768), (768, 1536), 2352),
    ((1, 117, 1536), (1536, 768), 1176),
    ((1, 29, 32, 128), (128, 256), 594),
    ((1, 29, 32, 256), (256, 128), 297),
]


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


class Sampler:
    """1 kHz sysfs clock reader. Holds no device fd, so it cannot itself perturb the arm."""

    def __init__(self, node, path):
        self.node, self.path, self.stop = node, Path(path), False
        self.n, self.lo, self.hi, self.err = 0, 10 ** 9, 0, 0
        self.gap_ms = 0.0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.path.open("w") as fh:
            last = time.time()
            while not self.stop:
                v = read_aiclk(self.node)
                now = time.time()
                self.gap_ms = max(self.gap_ms, 1e3 * (now - last))
                last = now
                if isinstance(v, int):
                    self.n += 1
                    self.lo, self.hi = min(self.lo, v), max(self.hi, v)
                else:
                    self.err += 1
                fh.write("%.6f %s\n" % (now, v))
                time.sleep(0.001)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=3)

    def summary(self):
        return {"samples": self.n, "min": self.lo if self.n else None,
                "max": self.hi if self.n else None, "read_errors": self.err,
                "worst_gap_ms": round(self.gap_ms, 2)}


sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or HERE / ("rf3_bwladder_%s.json" % a.tag)

    import torch
    from tt_bio import tenstorrent as T

    clk_fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    before = read_aiclk(a.node)
    status = "0x%02X" % _smc(clk_fd, FORCE_AICLK, a.clock)
    atexit.register(lambda: _smc(clk_fd, FORCE_AICLK, 0))
    t0 = time.time()
    while read_aiclk(a.node) != a.clock and time.time() - t0 < 10.0:
        time.sleep(0.01)
    if read_aiclk(a.node) != a.clock:
        print("REFUSING: node%d at %s MHz" % (a.node, read_aiclk(a.node)))
        return 2
    print("node%d FORCE_AICLK(%d) status=%s before=%s, locked after %.0f ms"
          % (a.node, a.clock, status, before, 1e3 * (time.time() - t0)), flush=True)

    with Sampler(a.node, out.with_suffix(".clk.jsonl")) as clk:
        rc = run(a, out, torch, T, clk)
    return rc


def run(a, out, torch, T, clk):
    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    # The shared kernel config every tt-bio module builds (tenstorrent.py:10979) and therefore the
    # one RF3's DiT projections run under.
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    print("grid %dx%d, CORE_GRID_MAIN %dx%d"
          % (g.x, g.y, T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y), flush=True)

    res = {"env": {"node": a.node, "clock": a.clock, "grid": [g.x, g.y], "tag": a.tag,
                   "reps": a.reps, "t_open": time.time(), "ttnn": ttnn.__file__,
                   "in1_block_tiles": T._MM_IN1_BLOCK_TILES,
                   "loadavg_open": Path("/proc/loadavg").read_text().split()[0]},
           "controls": {}, "groups": []}

    n = 8192
    f1 = 2.0 * n ** 3
    f2 = float((n // 32) ** 3) * 2 * 32 ** 3
    assert f1 == f2 == 1099511627776.0, (f1, f2)
    A = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=DRAM)
    B = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=DRAM)
    ms = []
    for i in range(6):
        ttnn.synchronize_device(device)
        t = time.perf_counter()
        r = ttnn.matmul(A, B, compute_kernel_config=KC, memory_config=DRAM)
        ttnn.synchronize_device(device)
        if i >= 2:
            ms.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(r)
    tf = f1 / 1e12 / (st.median(ms) / 1e3)
    res["controls"]["cube8192"] = {"flops_route1": f1, "flops_route2": f2,
                                   "tflops": round(tf, 3), "band": [112, 128],
                                   "pass": 112 <= tf <= 128}
    print("CONTROL cube8192 %.3f TFLOP/s pass=%s" % (tf, res["controls"]["cube8192"]["pass"]),
          flush=True)
    ttnn.deallocate(A)
    ttnn.deallocate(B)

    nb = 3 * 8192 * 8192 * 2
    assert nb == 402653184
    C = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=DRAM)
    D = ttnn.from_torch(torch.randn(8192, 8192, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=DRAM)
    ms = []
    for i in range(8):
        ttnn.synchronize_device(device)
        t = time.perf_counter()
        r = ttnn.add(C, D, memory_config=DRAM)
        ttnn.synchronize_device(device)
        if i >= 2:
            ms.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(r)
    gbs = nb / 1e9 / (st.median(ms) / 1e3)
    res["controls"]["dram_add"] = {"bytes": nb, "gbs": round(gbs, 2), "band": [400, 460],
                                   "pass": 400 <= gbs <= 460}
    print("CONTROL dram_add %.2f GB/s pass=%s" % (gbs, res["controls"]["dram_add"]["pass"]),
          flush=True)
    ttnn.deallocate(C)
    ttnn.deallocate(D)

    for xs, ws, calls in RF3_GROUPS:
        x = ttnn.from_torch(torch.randn(*xs, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(*ws, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=DRAM)
        # The arm is the shipped decision path: flip the module flag through its own setter and
        # ask `_short_m_proj_config` for this operand pair. If it declines, the group is an A/A
        # and says so.
        prev = T.set_mm_short_m_bw(True)
        before_stats = list(T.MM_SHORT_M_STATS)
        pc = T._short_m_proj_config(x, w)
        served = [b - c for b, c in zip(T.MM_SHORT_M_STATS, before_stats)]
        T.set_mm_short_m_bw(prev)
        grp = {"x": list(xs), "w": list(ws), "calls_per_fold": calls,
               "served": served[0], "declined": served[1]}
        if pc is None:
            grp["skipped"] = "rule declines this shape"
            res["groups"].append(grp)
            print("DECLINED %s x %s" % (xs, ws), flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
            continue
        grp["cfg"] = {"in0_block_w": pc.in0_block_w, "per_core_M": pc.per_core_M,
                      "per_core_N": pc.per_core_N, "out_subblock_h": pc.out_subblock_h,
                      "out_subblock_w": pc.out_subblock_w}
        out_bytes = 1
        for d in xs[:-1]:
            out_bytes *= d
        out_bytes *= ws[-1] * 2
        chain = max(1, min(8, int(2.5e8 // max(out_bytes, 1))))
        grp["chain"] = chain

        def run_arm(arm):
            routing = ({"program_config": pc} if arm == "bw"
                       else {"core_grid": T.CORE_GRID_MAIN})
            outs = [ttnn.linear(x, w, compute_kernel_config=KC, **routing)
                    for _ in range(chain)]
            return outs

        arms = {"base": [], "bw": [], "aa": []}
        order = ["base", "bw", "aa"]
        for _ in range(a.warm):
            for arm in order:
                for o in run_arm(arm):
                    ttnn.deallocate(o)
        for rep in range(a.reps):
            seq = order if rep % 2 == 0 else order[::-1]
            for arm in seq:
                ttnn.synchronize_device(device)
                t = time.perf_counter()
                outs = run_arm(arm)
                ttnn.synchronize_device(device)
                arms[arm].append((time.perf_counter() - t) * 1e6 / chain)
                for o in outs:
                    ttnn.deallocate(o)

        vals = {}
        tb = ttnn.to_torch(run_arm("base")[0]).float()
        tw = ttnn.to_torch(run_arm("bw")[0]).float()
        vals["equal"] = bool(torch.equal(tb, tw))
        vals["max_abs"] = float((tb - tw).abs().max().item())
        grp["values"] = vals

        for arm in order:
            grp[arm + "_us"] = round(st.median(arms[arm]), 3)
            grp[arm + "_spread"] = round((max(arms[arm]) - min(arms[arm]))
                                         / st.median(arms[arm]), 4)
        grp["us"] = {k: [round(v, 3) for v in vv] for k, vv in arms.items()}
        # Paired rep by rep: the arms share a rep, so pairing removes the drift the session has.
        pr = [arms["bw"][i] / arms["base"][i] for i in range(a.reps)]
        aa = [arms["aa"][i] / arms["base"][i] for i in range(a.reps)]
        grp["ratio_bw_paired"] = round(st.median(pr), 4)
        grp["ratio_aa_paired"] = round(st.median(aa), 4)
        grp["aa_halfwidth"] = round(abs(st.median(aa) - 1.0), 4)
        grp["resolved"] = abs(grp["ratio_bw_paired"] - 1.0) > 2 * max(grp["aa_halfwidth"], 1e-9)
        res["groups"].append(grp)
        print("%-22s x %-14s bw=%d  base %8.3f us  bw %8.3f us  ratio %.4f  A/A %.4f  %s"
              % (xs, ws, pc.in0_block_w, grp["base_us"], grp["bw_us"],
                 grp["ratio_bw_paired"], grp["ratio_aa_paired"],
                 "REGRESSION" if grp["ratio_bw_paired"] > 1.0 else "win"), flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)

    res["clock"] = clk.summary()
    res["env"]["t_close"] = time.time()
    res["env"]["aiclk_close"] = read_aiclk(a.node)
    res["env"]["loadavg_close"] = Path("/proc/loadavg").read_text().split()[0]
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out, flush=True)
    print("CLOCK %s" % res["clock"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
