#!/usr/bin/env python3
"""Per-call time versus core count, on Blackhole, at the fold's own 512 aa shapes.

Nobody has measured a Blackhole core-count sweep for these classes. The lever ledger's
"per-class grid sizing" entry came from a Wormhole 8x9 sweep on which the triangle product wanted
32 cores (2.10x) and the SDPA wanted 72 (1.36x the other way) -- opposite signs from one session,
so the sign is per class and does not transfer across architectures either.

The re-scoped question is a mechanism, not a ratio. `percall_residual/` found the matmul gap flat
across a 1152x FLOP range at a median 22.4 us per call, 13x the device's 1.70 us program-launch
floor. A flat per-call cost has two candidate producers, and a core-count sweep separates them:

  occupancy        per-call time falls as cores rise, then saturates; the knee moves with tile count
  per-program cost per-call time is roughly flat in core count and the sweep finds nothing

Both answers are a full pass. The second eliminates the cheaper explanation.

Three controls, because a sweep that finds nothing must be shown to be able to find something:

  cube       a dense 4096^3. It MUST scale with cores. If it does not, the instrument is broken and
             no flat curve in this session means anything.
  A/A        the 11x10 grid measured twice per rep, in different ladder positions, as this
             session's own noise floor rather than a borrowed one.
  enqueue    every region records host issue time as well as wall time. If they coincide the region
             is host-bound and its "per-call cost" is a Python dispatch cost, not a device one
             (`roof-launch-floor-per-device-program-not-python-call`).

Arms interleave per rep and the grid ladder is the inner loop, so drift and warmup land on every
grid equally (`op-ab-must-interleave-arms-compile-warmup-bias`). AICLK is forced to 1350 MHz for
the session and sampled DURING every timed region; a region whose samples are not all 1350 is
recorded and dropped, not reported.

No production code is changed. TT_BIO_FORCE_GRID is a shipped default-off knob, and the sweep
restores the measured grid before exiting.
"""
import argparse, fcntl, json, os, struct, subprocess, sys, time
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
from tt_bio import tenstorrent as T                                           # noqa: E402

IOC = (0xFA << 8) | 17
POST, POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33
TARGET_MHZ = 1350

# (x, y). The device grid is 11x10; the rest is the brief's ladder, kept rectangular so every
# entry is a grid ttnn will accept for all the shapes here (verified by probe.py).
LADDER = [(11, 10), (9, 8), (8, 8), (8, 6), (8, 4), (6, 4), (4, 4)]
AA_GRID = (11, 10)          # measured a second time, late in the ladder, as the noise floor


def smc(fd, mt, *args):
    msg = [mt] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, IOC, struct.pack("=IIII8I", 48, POST, 0, 0, *msg))
    dl = time.time() + 2.0
    while time.time() < dl:
        buf = bytearray(struct.pack("=IIII8I", 48, POLL, 0, 0, *([0] * 8)))
        try:
            fcntl.ioctl(fd, IOC, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005)
                continue
            raise
        r = struct.unpack("=IIII8I", bytes(buf))[4:]
        return r[0] & 0xFF, r[0] >> 16
    raise TimeoutError("no ARC response")


def coverage(samples, t0_ns, t1_ns):
    """Clock verdict for one timed region. Only reads fully inside the region count."""
    during = [s for s in samples
              if s.get("read_start_ns", -1) >= t0_ns and s.get("read_end_ns", t1_ns + 1) <= t1_ns]
    valid = [s for s in during if "MHz" in s]
    centers = [(s["read_start_ns"] + s["read_end_ns"]) // 2 for s in valid]
    pts = [t0_ns] + centers + [t1_ns]
    r = {"samples": len(valid),
         "min_MHz": min((s["MHz"] for s in valid), default=None),
         "max_MHz": max((s["MHz"] for s in valid), default=None),
         "max_gap_ms": max(b - a for a, b in zip(pts, pts[1:])) / 1e6,
         "W_max": max((s["W"] for s in valid if "W" in s), default=None),
         "errors": len([s for s in during if "error" in s])}
    r["pass"] = (r["samples"] >= 8 and r["min_MHz"] == r["max_MHz"] == TARGET_MHZ
                 and r["max_gap_ms"] <= 25.0 and r["errors"] == 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--region-ms", type=float, default=120.0)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    S = a.n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = T.get_device()
    dg = dev.compute_with_storage_grid_size()
    assert str(dev.arch()) == "Arch.BLACKHOLE", f"this row measures Blackhole, got {dev.arch()}"
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    root = Path(f"/sys/class/tenstorrent/tenstorrent!{a.node}")
    fd = os.open(f"/dev/tenstorrent/{a.node}", os.O_RDWR | os.O_APPEND)
    force = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
    if force[0] != 0:
        raise RuntimeError(f"FORCE_AICLK({TARGET_MHZ}) refused: {force}")
    clk_path = out / "clock.jsonl"
    sampler = subprocess.Popen(
        [sys.executable, str(HERE / "clocksample.py"), str(clk_path), str(a.node)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(0.5)

    def dram(t):
        return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    g = torch.Generator().manual_seed(0)

    def rn(*s):
        return torch.randn(*s, generator=g) * 0.1

    # -- the fold's own matmul shapes, call counts from perf/roof_launch/op_census_512.json via
    #    perf/c10_orchestrator/percall_residual/percall_residual.json --------------------------
    MM = {
        # label            A                        B                  calls   family
        "mm_768_1536":  (rn(1, S, 768),  rn(768, 1536),  11200 + 4000, "768-family linear"),
        "mm_768_768":   (rn(1, S, 768),  rn(768, 768),
                         10200 + 9200 + 6600 + 4200 + 3800 + 1800 + 1600, "768-family linear"),
        "mm_1536_768":  (rn(1, S, 1536), rn(1536, 768),  4800, "768-family linear"),
        "mm_768_3072":  (rn(1, S, 768),  rn(768, 3072),  3000 + 1800, "768-family linear"),
        "mm_pair_qk":   (rn(1, 16, S, 128), rn(128, 512), 8448, "16-head pair matmul"),
        "mm_pair_av":   (rn(1, 16, S, 512), rn(512, 128), 8448, "16-head pair matmul"),
    }
    mm = {k: (dram(A), dram(B), calls, fam) for k, (A, B, calls, fam) in MM.items()}
    cube_a, cube_b = dram(rn(4096, 4096)), dram(rn(4096, 4096))

    # -- the two tri classes, at the fold's 512 aa shape ---------------------------------------
    def tm_weights(cz=128, hidden=128):
        return {"norm_in.weight": torch.ones(cz), "norm_in.bias": torch.zeros(cz),
                "norm_out.weight": torch.ones(hidden), "norm_out.bias": torch.zeros(hidden),
                "g_in.weight": rn(2 * hidden, cz), "p_in.weight": rn(2 * hidden, cz),
                "g_out.weight": rn(cz, cz), "p_out.weight": rn(cz, hidden)}

    def ta_weights(c_z=128, n_heads=4, head_dim=32):
        d = n_heads * head_dim
        return {"layer_norm.weight": torch.ones(c_z), "layer_norm.bias": torch.zeros(c_z),
                "linear_q.weight": rn(d, c_z), "linear_k.weight": rn(d, c_z),
                "linear_v.weight": rn(d, c_z), "linear_g.weight": rn(d, c_z),
                "linear_o.weight": rn(c_z, d), "linear.weight": rn(n_heads, c_z)}

    TMW, TAW = tm_weights(), ta_weights()
    z4 = dram(rn(1, S, S, 128))
    s_ = (torch.rand(1, S, generator=g) > 0.25).float()
    mask = dram(s_[:, :, None] * s_[:, None, :])

    def set_grid(gx, gy):
        os.environ["TT_BIO_FORCE_GRID"] = f"{gx},{gy}"
        T._configure_active_compute_grid(dev)
        assert tuple(T.COMPUTE_GRID_MAIN) == (gx, gy), T.COMPUTE_GRID_MAIN

    units = {}

    def build_units():
        units["trimul"] = T.TriangleMultiplication(False, TMW, ckc)
        units["triatt"] = T.TriangleAttention(32, 4, False, TAW, ckc)

    # -- arms ----------------------------------------------------------------------------------
    def mk_mm(label):
        A, B, _c, _f = mm[label]

        def fn(gx, gy):
            ttnn.deallocate(ttnn.linear(A, B, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                        core_grid=ttnn.CoreGrid(y=gy, x=gx)))
        return fn

    def f_cube(gx, gy):
        ttnn.deallocate(ttnn.matmul(cube_a, cube_b, compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    core_grid=ttnn.CoreGrid(y=gy, x=gx)))

    def f_trimul(gx, gy):
        ttnn.deallocate(units["trimul"](z4, mask))

    def f_triatt(gx, gy):
        ttnn.deallocate(units["triatt"](z4))

    ARMS = {}
    ARMS["cube"] = {"fn": f_cube, "flops": 2 * 4096 ** 3, "calls": None,
                    "family": "control: dense cube, must scale with cores",
                    "grid_mode": "arg"}
    for label, (_A, _B, calls, fam) in mm.items():
        M, K, N = MM[label][0].shape[-2], MM[label][0].shape[-1], MM[label][1].shape[-1]
        b = 1
        for d in MM[label][0].shape[:-2]:
            b *= d
        ARMS[label] = {"fn": mk_mm(label), "flops": 2 * b * M * K * N, "calls": calls,
                       "family": fam, "grid_mode": "arg",
                       "shape": f"out={b}x{M}x{N} in={b}x{M}x{K},{K}x{N}"}
    TM_F = 2 * S * S * 128 * 5 * 128 + 2 * 128 * S ** 3 + 2 * S * S * 128 * 128
    TA_F = (2 * S * S * 128 * (3 * 128 + 128 + 32) + 2 * 2 * (S * 4) * S * S * 32
            + 2 * S * S * 128 * 128)
    ARMS["trimul"] = {"fn": f_trimul, "flops": TM_F, "calls": 528,
                      "family": "shipped TriangleMultiplication unit", "grid_mode": "global"}
    ARMS["triatt"] = {"fn": f_triatt, "flops": TA_F, "calls": 528,
                      "family": "shipped TriangleAttention unit", "grid_mode": "global"}

    order = [(g_, f"{g_[0]}x{g_[1]}") for g_ in LADDER]
    order.insert(len(order) - 1, (AA_GRID, f"{AA_GRID[0]}x{AA_GRID[1]}#AA"))

    meta = {"n": S, "arch": str(dev.arch()), "device_grid": [int(dg.x), int(dg.y)],
            "host": os.uname().nodename, "node": a.node,
            "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
            "loadavg_start": os.getloadavg(), "reps": a.reps, "warm": a.warm,
            "region_ms_target": a.region_ms, "force_response": force,
            "ladder": [k for _g, k in order], "target_MHz": TARGET_MHZ,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "srcversion": Path("/sys/module/tenstorrent/srcversion").read_text().strip(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(meta), flush=True)

    # -- size each arm's repetition count once, at the full grid --------------------------------
    set_grid(*AA_GRID)
    build_units()
    R = {}
    for name, arm in ARMS.items():
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        arm["fn"](*AA_GRID)
        ttnn.synchronize_device(dev)
        one = time.perf_counter() - t0
        R[name] = max(4, min(20000, int(round(a.region_ms / 1e3 / max(one, 1e-6)))))
        print(f"size {name}: first call {1e3*one:.3f} ms -> R={R[name]}", flush=True)
    meta["reps_per_region"] = R

    rows = []
    for i in range(a.warm + a.reps):
        for name, arm in ARMS.items():
            for (gx, gy), key in order:
                if arm["grid_mode"] == "global":
                    set_grid(gx, gy)
                    build_units()
                n = R[name]
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                t0n = time.monotonic_ns()
                try:
                    for _ in range(n):
                        arm["fn"](gx, gy)
                    tq = time.perf_counter()
                    ttnn.synchronize_device(dev)
                except Exception as e:
                    ttnn.synchronize_device(dev)
                    rows.append({"rep": i, "warm": i < a.warm, "arm": name, "grid": key,
                                 "cores": gx * gy, "R": n,
                                 "refused": f"{type(e).__name__}: {str(e)[:200]}"})
                    print(f"  refused {name}@{key}: {type(e).__name__}", flush=True)
                    continue
                t1 = time.perf_counter()
                t1n = time.monotonic_ns()
                # Host issue cost with an EMPTY queue. In the region above the host blocks on a
                # full command queue once the device is the slower side, so enqueue_s there
                # cannot tell host-bound from device-bound. A short burst can: nothing has had
                # time to fill, so this is the pure Python-side cost of issuing one call.
                ttnn.synchronize_device(dev)
                tb = time.perf_counter()
                for _ in range(8):
                    arm["fn"](gx, gy)
                issue_us = 1e6 * (time.perf_counter() - tb) / 8
                ttnn.synchronize_device(dev)
                rows.append({"rep": i, "warm": i < a.warm, "arm": name, "grid": key,
                             "issue_us": issue_us,
                             "cores": gx * gy, "gx": gx, "gy": gy, "R": n,
                             "wall_s": t1 - t0, "enqueue_s": tq - t0,
                             "per_call_us": 1e6 * (t1 - t0) / n,
                             "enqueue_us": 1e6 * (tq - t0) / n,
                             "t0_ns": t0n, "t1_ns": t1n})
            if arm["grid_mode"] == "global":
                set_grid(*AA_GRID)
                build_units()
        done = [r for r in rows if r["rep"] == i]
        print("rep %d  %s" % (i, "  ".join(
            "%s@%s %.1fus" % (r["arm"], r["grid"], r["per_call_us"])
            for r in done[:8] if "per_call_us" in r)), flush=True)

    # -- stop the sampler, join the clock to every region ---------------------------------------
    try:
        sampler.communicate(b"stop\n", timeout=20)
    except subprocess.TimeoutExpired:
        sampler.terminate(); sampler.wait(timeout=5)
    samples = [json.loads(l) for l in clk_path.read_text().splitlines() if l.strip()]
    for r in rows:
        if "t0_ns" in r:
            r["clock"] = coverage(samples, r["t0_ns"], r["t1_ns"])

    rel = list(smc(fd, FORCE_AICLK, 0))
    os.close(fd)
    os.environ.pop("TT_BIO_FORCE_GRID", None)
    T._configure_active_compute_grid(dev)

    meta["release_response"] = rel
    meta["loadavg_end"] = os.getloadavg()
    meta["clock_samples"] = len(samples)
    meta["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "fn"} for k, v in ARMS.items()}
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = {"meta": meta, "rows": rows}
    (out / "sweep.json").write_text(json.dumps(res, indent=1))
    bad = [r for r in rows if not r["warm"] and not r.get("clock", {}).get("pass")]
    print(f"\nwrote {out/'sweep.json'}  rows={len(rows)}  clock-rejected(scored)={len(bad)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
