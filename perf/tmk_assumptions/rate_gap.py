#!/usr/bin/env python3
"""tmk-assumptions: re-measure the campaign's load-bearing rate claims.

The campaign's premise is a 2.2-2.9x "kernel-rate deficit": three trimul call sites measured at
54.97 / 42.20 / 43.34 TFLOP/s against a 123.65 TFLOP/s square-compute roof. A TFLOP/s gap is only
headroom if the SHAPE can reach it (roof-headroom-needs-arithmetic-intensity-not-just-tflops-gap),
so this script measures, in one session at one pinned and during-sampled clock:

  * the square compute roof, in the shipped kernel config (a mismatched config is worth 1.40x);
  * the DRAM roof at THREE read:write mixes, because each call site moves a different mix and a
    combined roof is not the roof of a 1R:2W stream;
  * the three production call sites at their production shapes and program configs;
  * for each, closed-form compulsory bytes -> arithmetic intensity -> the roof that actually binds.

Every per-call time comes from an n-ladder slope, never a single synced bracket: a `sync; call;
sync` bracket charges a ~0.05 ms host floor that is half of some of these calls
(synced-bracket-inflates-op-level-fixed-cost).
"""
import argparse, json, os, statistics, subprocess, sys, threading, time
from pathlib import Path

import torch
import ttnn

BF16 = 2  # bytes per element


# ---------------------------------------------------------------- clock

class ClockSampler:
    """Dense AICLK sampling straight off sysfs. Holds no device fd, so it cannot be taken away
    with the device, and its span is recorded so it can be checked against the measurement's span
    (c13 lesson 2: a record that certifies less than it covers is not evidence)."""

    def __init__(self, node, period=0.05):
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk")
        self.period, self.samples, self.errors = period, [], 0
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), int(self.path.read_text())))
            except Exception:
                self.errors += 1
            time.sleep(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self

    def __exit__(self, *e):
        self._stop.set(); self._t.join(timeout=5)

    def read(self):
        try:
            return int(self.path.read_text())
        except Exception:
            return None

    def summary(self, t0, t1):
        win = [c for (t, c) in self.samples if t0 <= t <= t1]
        ts = [t for (t, _) in self.samples if t0 <= t <= t1]
        gaps = [b - a for a, b in zip(ts, ts[1:])] or [0.0]
        return dict(n=len(win), min=min(win) if win else None, max=max(win) if win else None,
                    median=statistics.median(win) if win else None,
                    max_gap_ms=round(max(gaps) * 1e3, 1), errors=self.errors,
                    sampler_span_s=round((ts[-1] - ts[0]) if len(ts) > 1 else 0.0, 2),
                    measure_span_s=round(t1 - t0, 2))


def pin_clock(node, target, repo):
    p = subprocess.Popen([sys.executable, str(repo / "perf/pvxcust/pin_aiclk.py"),
                          str(node), str(target)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for _ in range(40):
        line = p.stdout.readline()
        if not line:
            break
        print("  pin:", line.rstrip(), flush=True)
        if line.startswith("READY"):
            break
    # FORCE_AICLK returns ~85 ms before the clock actually leaves the governor's value, so wait
    # for the ARC to arrive rather than trusting the return.
    sysfs = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk")
    for _ in range(200):
        if int(sysfs.read_text()) >= target - 10:
            break
        time.sleep(0.05)
    print(f"  pin: aiclk now {int(sysfs.read_text())} MHz", flush=True)
    return p


# ---------------------------------------------------------------- timing

def ladder(dev, fn, ns=(1, 2, 4, 8), reps=5, serial=False):
    """Per-call ms from the slope of t(n), plus the intercept the bracket charges.

    serial=True syncs after every call (the class-rate convention trix-floor used); serial=False
    enqueues n back to back and syncs once, which is what a fold actually sees."""
    for _ in range(2):                      # warm: JIT + program cache
        fn(); ttnn.synchronize_device(dev)
    per_n = {}
    for n in ns:
        best = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
                if serial:
                    ttnn.synchronize_device(dev)
            ttnn.synchronize_device(dev)
            best.append(time.perf_counter() - t0)
        per_n[n] = statistics.median(best)
    lo, hi = min(ns), max(ns)
    slope = (per_n[hi] - per_n[lo]) / (hi - lo)
    return dict(ms=slope * 1e3, intercept_ms=(per_n[lo] - slope * lo) * 1e3,
                raw={str(k): round(v * 1e3, 4) for k, v in per_n.items()})


def dram(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="perf/tmk_assumptions/rate_gap.json")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from tt_bio import tenstorrent as T
    # A lone P300 chip is a CUSTOM cluster topology and `ttnn.open_device` hard-fatals without a
    # 1x1 Blackhole mesh graph descriptor. Production sets it per worker; a bare harness has no
    # shard to inherit it from, so call the shipped helper.
    from tt_bio.main import ensure_p300_mesh_descriptor
    print("  mgd:", ensure_p300_mesh_descriptor(), flush=True)

    pin = pin_clock(a.node, a.clock, repo)
    clk = ClockSampler(a.node)
    res, arms = {}, {}
    t_start = time.time()
    with clk:
        dev = ttnn.open_device(device_id=0)
        try:
            T.CORE_GRID_MAIN = dev.core_grid
            T.COMPUTE_GRID_MAIN = (dev.core_grid.x, dev.core_grid.y)
            gx, gy = T.COMPUTE_GRID_MAIN
            print(f"  grid {gx}x{gy}", flush=True)
            ckc = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)

            def run(name, fn, flop, byts, note=""):
                s = ladder(dev, fn, reps=a.reps, serial=True)
                p = ladder(dev, fn, reps=a.reps, serial=False)
                arms[name] = dict(
                    serial_ms=round(s["ms"], 5), pipe_ms=round(p["ms"], 5),
                    serial_intercept_ms=round(s["intercept_ms"], 5),
                    serial_raw=s["raw"], pipe_raw=p["raw"],
                    GFLOP=round(flop / 1e9, 4), MB=round(byts / 1e6, 3),
                    AI_flop_per_byte=round(flop / byts, 2) if byts else None,
                    serial_TFLOPs=round(flop / (s["ms"] * 1e-3) / 1e12, 3) if flop else None,
                    pipe_TFLOPs=round(flop / (p["ms"] * 1e-3) / 1e12, 3) if flop else None,
                    serial_GBs=round(byts / (s["ms"] * 1e-3) / 1e9, 2),
                    pipe_GBs=round(byts / (p["ms"] * 1e-3) / 1e9, 2), note=note)
                print(f"  {name:26s} serial {s['ms']:8.4f} ms  pipe {p['ms']:8.4f} ms  "
                      f"{arms[name]['serial_TFLOPs'] or 0:7.2f} TF/s  "
                      f"{arms[name]['serial_GBs']:7.1f} GB/s  AI {arms[name]['AI_flop_per_byte']}",
                      flush=True)
                return arms[name]

            # ---- roofs ------------------------------------------------------
            print("ROOFS", flush=True)
            c = 4096
            ca = dram(torch.randn(c, c, dtype=torch.bfloat16), dev)
            cb = dram(torch.randn(c, c, dtype=torch.bfloat16), dev)
            run("roof_cube4096", lambda: ttnn.matmul(ca, cb, compute_kernel_config=ckc,
                                                     dtype=ttnn.bfloat16),
                2 * c ** 3, 3 * c * c * BF16, "shipped ckc HiFi4/fp32T/pl1T")
            run("roof_cube4096_AA", lambda: ttnn.matmul(ca, cb, compute_kernel_config=ckc,
                                                        dtype=ttnn.bfloat16),
                2 * c ** 3, 3 * c * c * BF16, "A/A twin")
            ca.deallocate(); cb.deallocate()

            # DRAM roofs at the three mixes the call sites actually move. Z-scale operands so the
            # roof is read at the same footprint as the arms.
            n2 = 8192
            za = dram(torch.randn(n2, n2, dtype=torch.bfloat16), dev)
            zb = dram(torch.randn(n2, n2, dtype=torch.bfloat16), dev)
            Zb = n2 * n2 * BF16
            run("roof_dram_2R1W", lambda: ttnn.add(za, zb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, 3 * Zb, "add: 2 reads 1 write")
            run("roof_dram_2R1W_AA", lambda: ttnn.add(za, zb,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, 3 * Zb, "A/A twin")
            run("roof_dram_1R1W", lambda: ttnn.multiply(za, 2.0,
                                                        memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, 2 * Zb, "scalar mul: 1 read 1 write")
            run("roof_dram_1R2W", lambda: ttnn.concat([za, za], dim=-1,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, 3 * Zb, "concat dup: 1 read 2 writes (a data-movement op, a weak roof)")
            run("roof_dram_1R0W", lambda: ttnn.sum(za), 0, Zb, "sum: read only")
            run("roof_dram_clone", lambda: ttnn.clone(za, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, 2 * Zb, "clone: 1 read 1 write")
            za.deallocate(); zb.deallocate()

            # ---- call sites -------------------------------------------------
            print("CALL SITES (N=512 tokens, D=256 channels)", flush=True)
            N, D, C = 512, 256, 128      # C = the shipped channel chunk
            P = N * N                    # 262144 pair rows

            # A: the in-projection, the production function, fused [g_a|g_b|p_a|p_b] at chunk C.
            xin = dram(torch.randn(1, 1, P, D, dtype=torch.bfloat16), dev)
            win = dram(torch.randn(D, 4 * C, dtype=torch.bfloat16), dev)
            fA = 2 * P * D * (4 * C)
            bA = P * D * BF16 + P * 4 * C * BF16 + D * 4 * C * BF16
            run("A_inproj_DRAM", lambda: T._in_proj_matmul(xin, win, ckc,
                                                           ttnn.DRAM_MEMORY_CONFIG),
                fA, bA, "_in_proj_matmul, [1,1,262144,256]@[256,512], DRAM out")
            run("A_inproj_DRAM_AA", lambda: T._in_proj_matmul(xin, win, ckc,
                                                              ttnn.DRAM_MEMORY_CONFIG),
                fA, bA, "A/A twin")
            xin.deallocate(); win.deallocate()

            # B: the contraction, production program config, both transpose_b legs.
            ma = dram(torch.randn(1, C, N, N, dtype=torch.bfloat16), dev)
            mb = dram(torch.randn(1, C, N, N, dtype=torch.bfloat16), dev)
            pc = T._triangle_mul_program_config((N + 31) // 32)
            fB = 2 * C * N * N * N
            bB = 3 * C * N * N * BF16
            run("B_contract_DRAM", lambda: ttnn.matmul(
                ma, mb, program_config=pc, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG), fB, bB,
                "[1,128,512,512]@[1,128,512,512], _triangle_mul_program_config(16)")
            run("B_contract_DRAM_AA", lambda: ttnn.matmul(
                ma, mb, program_config=pc, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG), fB, bB, "A/A twin")
            run("B_contract_tb", lambda: ttnn.matmul(
                ma, mb, transpose_b=True, program_config=pc, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG), fB, bB,
                "transpose_b=True, the shipped deferred-transpose leg")
            ma.deallocate(); mb.deallocate()

            # C: the output projection.
            xo = dram(torch.randn(1, N, N, D, dtype=torch.bfloat16), dev)
            wo = dram(torch.randn(D, D, dtype=torch.bfloat16), dev)
            fC = 2 * P * D * D
            bC = P * D * BF16 * 2 + D * D * BF16
            run("C_outproj_linear", lambda: T._pair_proj_linear(xo, wo, ckc, ttnn.bfloat16),
                fC, bC, "_pair_proj_linear, [1,512,512,256]@[256,256], DRAM out")
            run("C_outproj_linear_AA", lambda: T._pair_proj_linear(xo, wo, ckc, ttnn.bfloat16),
                fC, bC, "A/A twin")
            run("C_outproj_minimal", lambda: ttnn.experimental.minimal_matmul(
                xo, wo, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc), fC, bC, "minimal_matmul, the _TRIMUL_MM_OUT leg")
            xo.deallocate(); wo.deallocate()

            # ---- the precision question --------------------------------------
            # FLOOR-3Z says "3Z is the floor and it does not depend on precision", which the
            # campaign reads as "bfp8 cannot help trimul". 3Z is a COUNT; the BYTES are
            # 3*N^2*D*width. If these sites are traffic-bound, halving the width must show up as
            # time, and that is the cleanest discriminator available: bfp8_b holds FLOPs exactly
            # constant and halves the bytes.
            print("BFP8_B: same FLOPs, 0.53x the bytes", flush=True)
            B8, W8 = ttnn.bfloat8_b, 1.0625
            xin8 = dram(torch.randn(1, 1, P, D, dtype=torch.bfloat16), dev, B8)
            win8 = dram(torch.randn(D, 4 * C, dtype=torch.bfloat16), dev, B8)
            bA8 = (P * D + P * 4 * C + D * 4 * C) * W8
            run("A_inproj_bfp8", lambda: ttnn.experimental.minimal_matmul(
                xin8, win8, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8,
                compute_kernel_config=ckc), fA, bA8, "bfp8_b operands and result")
            xin8.deallocate(); win8.deallocate()

            ma8 = dram(torch.randn(1, C, N, N, dtype=torch.bfloat16), dev, B8)
            mb8 = dram(torch.randn(1, C, N, N, dtype=torch.bfloat16), dev, B8)
            run("B_contract_bfp8", lambda: ttnn.matmul(
                ma8, mb8, program_config=pc, compute_kernel_config=ckc, dtype=B8,
                memory_config=ttnn.DRAM_MEMORY_CONFIG), fB, 3 * C * N * N * W8,
                "bfp8_b operands and result")
            ma8.deallocate(); mb8.deallocate()

            xo8 = dram(torch.randn(1, N, N, D, dtype=torch.bfloat16), dev, B8)
            wo8 = dram(torch.randn(D, D, dtype=torch.bfloat16), dev, B8)
            run("C_outproj_bfp8", lambda: T._pair_proj_linear(xo8, wo8, ckc, B8),
                fC, (2 * P * D + D * D) * W8, "bfp8_b operands and result")
            xo8.deallocate(); wo8.deallocate()

            # ---- GRANULE: what the channel move actually costs, by route -------
            # The claim is that the pair-major -> channel-major exchange is compulsory at a 64 B
            # granule. Price the shipped route against two routes that ARE tile-granular on the
            # same bytes, so the granule's cost is a measured difference rather than an assertion.
            print("GRANULE: the pair-major -> channel-major exchange, by route", flush=True)
            zc = dram(torch.randn(1, N, N, C, dtype=torch.bfloat16), dev)
            mv = N * N * C * BF16 * 2
            run("G_channel_move_0312", lambda: ttnn.permute(zc, (0, 3, 1, 2)), 0, mv,
                "the SHIPPED channel move: [1,512,512,128] -> [1,128,512,512], 64 B granule")
            run("G_channel_move_0312_AA", lambda: ttnn.permute(zc, (0, 3, 1, 2)), 0, mv, "A/A twin")
            run("G_channel_move_0321", lambda: ttnn.permute(zc, (0, 3, 2, 1)), 0, mv,
                "the other role's move")
            zc.deallocate()
            zt = dram(torch.randn(1, C, N, N, dtype=torch.bfloat16), dev)
            run("G_tile_transpose_last2", lambda: ttnn.transpose(zt, -2, -1), 0, mv,
                "TILE-granular control: same bytes, last-two-axis transpose")
            run("G_clone_same_bytes", lambda: ttnn.clone(zt, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                0, mv, "no-permute control: the same bytes copied")
            zt.deallocate()
        finally:
            ttnn.close_device(dev)
    t_end = time.time()

    res["clock"] = clk.summary(t_start, t_end)
    res["clock"]["target"] = a.clock
    res["arms"] = arms
    res["node"] = a.node
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["clock"], indent=1), flush=True)
    pin.terminate()
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
