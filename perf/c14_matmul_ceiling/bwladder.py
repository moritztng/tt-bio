#!/usr/bin/env python3
"""in0_block_w ladder on the fold's OWN starved matmul configs. One field varies, nothing else.

`bw_census.py` found 20 (shape, config) groups carrying 2.2313 s of the matmul class's 3.9459 s
that resolve to `in0_block_w=1` over a K of 2 to 140 tiles: they stream the inner dimension one
32x32 tile at a time. The fold itself contains the control -- `(1,512,768)x(768,768)` appears twice,
same grid, same per_core_M/N, same 64 cores, same DRAM operands and bias, differing only in
`fuse_batch` and therefore in the derived `in0_block_w`: 23,300 ns at 1 against 15,069 ns at 4.

So the arm here is not a hand-written config. `cfg_probe.py` reads the fold's resolved config out of
the committed capture and this bench varies exactly `in0_block_w`, keeping the factory, the grid,
the block schedule, the per-core blocking, the operand memory configs, the bias and the kernel
config as the fold has them. Session s1 is why: an arm that did not carry the fold's config read the
fold's own shape 2.2x slow and blew a pre-registered band.

Two things s1 also forces:
  * every timed region issues CHAIN calls, so the host round trip divides away (s1 measured 25.0 us
    of it against a 20.66 us kernel);
  * the per-shape control is the FOLD's own us/call at bw=1. An arm that cannot reproduce that
    cannot price a change to it.

KNOWN-ANSWER CONTROLS, same session, before any arm is believed: the 8192^3 cube under the arms'
own kernel config, FLOPs asserted equal by two independent routes, and the starved 8192^2 add of
exactly 402,653,184 bytes. Values are checked too, not just times: every bw arm's output is
compared against the bw=1 arm's on device-read tensors, and both `torch.equal` and the max absolute
deviation are reported, because a different K-blocking with packer_l1_acc accumulates partial sums
in a different order.
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


def divisors_upto(kt, cap):
    return [d for d in range(1, kt + 1) if kt % d == 0 and d <= cap]


def build_cfg(factory, cfg, grid, bw):
    g = ttnn.CoreCoord(grid[0], grid[1])
    # out_block_h/out_block_w size the drain schedule and therefore the output CB. Dropping them
    # is not a cosmetic omission: the first run of this bench left them at the factory default and
    # the device refused the OPM projection at EVERY rung with a static-CB clash, at a config the
    # fold runs 16 times per fold.
    common = dict(compute_with_storage_grid_size=g, in0_block_w=bw,
                  out_subblock_h=cfg["out_subblock_h"], out_subblock_w=cfg["out_subblock_w"],
                  out_block_h=cfg.get("out_block_h", cfg["per_core_M"]),
                  out_block_w=cfg.get("out_block_w", cfg["per_core_N"]),
                  per_core_M=cfg["per_core_M"], per_core_N=cfg["per_core_N"],
                  fuse_batch=bool(cfg.get("fuse_batch", 1)), fused_activation=None)
    if factory.endswith("MultiCast1DProgramConfig"):
        cls = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
        kw = dict(common, mcast_in0=bool(cfg.get("mcast_in0", 0)))
    else:
        cls = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
        kw = dict(common, transpose_mcast=bool(cfg.get("transpose_mcast", 0)))
    for drop in ([], ["fused_activation"], ["out_block_h", "out_block_w"],
                 ["fused_activation", "out_block_h", "out_block_w"]):
        try:
            return cls(**{k: v for k, v in kw.items() if k not in drop})
        except TypeError:
            continue
    raise TypeError("no accepted kwarg set for %s" % cls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=HERE / "fold_configs.json")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--min-fold-s", type=float, default=0.03)
    ap.add_argument("--max-shapes", type=int, default=8)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--tag", default="b1")
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T

    groups = [g for g in json.loads(a.configs.read_text())
              if g["in0_block_w"] == 1 and g["k_tiles"] > 1 and g["fold_s"] >= a.min_fold_s]
    groups = groups[:a.max_shapes]
    print("screening %d starved groups, %.4f s of fold time"
          % (len(groups), sum(g["fold_s"] for g in groups)), flush=True)

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
    print("node%d FORCE_AICLK(%d) status=%s before=%s, at %d MHz after %.0f ms"
          % (a.node, a.clock, status, before, a.clock, 1e3 * (time.time() - t0)), flush=True)

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    print("grid %dx%d" % (g.x, g.y), flush=True)

    res = {"env": {"node": a.node, "clock": a.clock, "grid": [g.x, g.y], "tag": a.tag,
                   "reps": a.reps, "t_open": time.time(), "ttnn": ttnn.__file__},
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

    for grp in groups:
        b, m, k, nn = grp["shape"]
        kt = grp["k_tiles"]
        cfg, fac, grid = grp["cfg"], grp["factory"], grp["grid"]
        mem0 = L1 if "L1" in grp["mem"][0] else DRAM
        mem1 = L1 if "L1" in grp["mem"][1] else DRAM
        memo = L1 if "L1" in grp["mem"][2] else DRAM
        out_bytes = b * m * nn * 2
        # An L1 destination is the binding budget, not a convenience: chaining 8 of a 24.6 MB L1
        # output asks for 197 MB against 160.79 MB of usable L1 and the DEVICE refuses the program
        # ("statically allocated circular buffers clash with L1 buffers"), which is what the first
        # run of this bench did on every rung of one group. The fold holds ONE such output live, so
        # the chain budget follows the destination.
        budget = 6.0e7 if memo is L1 else 2.5e8
        chain = max(1, min(8, int(budget // max(out_bytes, 1))))
        # in0 CB per core is per_core_M x bw tiles double buffered; keep it under 1 MB of the
        # 1.46 MB usable L1 so the ladder is refused by the DEVICE, not by this bench.
        cap = max(1, int(1.0e6 // (cfg["per_core_M"] * 2 * 2048)))
        bws = divisors_upto(kt, min(kt, cap))
        if len(bws) < 2:
            print("SKIP %s: kt=%d, per_core_M=%d leaves no wider block (cap %d)"
                  % (grp["shape"], kt, cfg["per_core_M"], cap), flush=True)
            res["groups"].append({**{kk: grp[kk] for kk in ("shape", "k_tiles", "fold_s",
                                                            "calls", "us_per_call", "cores")},
                                  "skipped": "no wider in0 block fits: cap %d" % cap})
            continue
        x = ttnn.from_torch(torch.randn(b, m, k, dtype=torch.bfloat16) if b > 1
                            else torch.randn(m, k, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem0)
        w = ttnn.from_torch(torch.randn(k, nn, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=mem1)
        bias = None
        if grp["bias"]:
            bias = ttnn.from_torch(torch.randn(1, nn, dtype=torch.bfloat16),
                                   layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

        def run(bw, keep=False):
            pc = build_cfg(fac, cfg, grid, bw)
            outs = []
            for _ in range(chain):
                outs.append(ttnn.linear(x, w, bias=bias, program_config=pc,
                                        compute_kernel_config=KC, memory_config=memo,
                                        dtype=ttnn.bfloat16))
            if keep:
                return outs
            for o in outs:
                ttnn.deallocate(o)
            return None

        arms = {}
        ref = None
        ok = True
        for bw in bws:
            try:
                run(bw)
                ttnn.synchronize_device(device)
            except Exception as e:                                            # noqa: BLE001
                arms[bw] = {"error": repr(e)[:180]}
                print("  bw=%-3d REFUSED %s" % (bw, repr(e)[:110]), flush=True)
                continue
            arms[bw] = {"ms": []}
        live = [bw for bw in bws if "error" not in arms[bw]]
        for _ in range(a.warm):
            for bw in live:
                run(bw)
        order = list(live)
        for rep in range(a.reps):
            seq = order if rep % 2 == 0 else order[::-1]
            for bw in seq:
                ttnn.synchronize_device(device)
                t = time.perf_counter()
                run(bw)
                ttnn.synchronize_device(device)
                arms[bw]["ms"].append((time.perf_counter() - t) * 1e3 / chain)
        # values: compare every arm against bw=1 on device-read tensors
        vals = {}
        base_t = None
        for bw in live:
            outs = run(bw, keep=True)
            t_ = ttnn.to_torch(outs[0]).float()
            for o in outs:
                ttnn.deallocate(o)
            if base_t is None:
                base_t = t_
                vals[bw] = {"equal": True, "max_abs": 0.0}
            else:
                d = (t_ - base_t).abs().max().item()
                vals[bw] = {"equal": bool(torch.equal(t_, base_t)), "max_abs": d,
                            "rel": d / max(base_t.abs().max().item(), 1e-9)}
        rows = {}
        for bw in live:
            msv = arms[bw]["ms"]
            med = st.median(msv)
            rows[bw] = {"us_per_call": round(med * 1e3, 3), "min_us": round(min(msv) * 1e3, 3),
                        "spread_pct": round(100 * (max(msv) - min(msv)) / med, 3),
                        "ratio_to_bw1": None, "values": vals.get(bw)}
        if not live:
            print("  REFUSED at every rung, group skipped", flush=True)
            res["groups"].append({"shape": grp["shape"], "k_tiles": kt, "fold_s": grp["fold_s"],
                                  "calls": grp["calls"], "skipped": "device refused every rung",
                                  "errors": {str(bw): arms[bw].get("error") for bw in bws}})
            ttnn.deallocate(x)
            ttnn.deallocate(w)
            if bias is not None:
                ttnn.deallocate(bias)
            continue
        base = rows[live[0]]["us_per_call"]
        for bw in live:
            rows[bw]["ratio_to_bw1"] = round(rows[bw]["us_per_call"] / base, 4)
        best = min(live, key=lambda bw: rows[bw]["us_per_call"])
        entry = {"shape": grp["shape"], "k_tiles": kt, "factory": fac, "cfg": cfg,
                 "mem": grp["mem"], "bias": grp["bias"], "cores": grp["cores"],
                 "chain": chain, "cap": cap, "bws": live,
                 "fold_us_per_call": grp["us_per_call"], "fold_s": grp["fold_s"],
                 "calls": grp["calls"], "rows": rows, "best_bw": best,
                 "bench_reproduces_fold": round(base / grp["us_per_call"], 4),
                 "fold_s_at_best": round(grp["calls"] * rows[best]["us_per_call"] / 1e6, 5)}
        # The two instruments are not interchangeable: the bench times a synced chain on an idle
        # L1, the fold's figure is a Tracy kernel duration inside a live block. So the RATIO comes
        # from the bench (one instrument, both arms, interleaved) and the BASE comes from the fold,
        # and a group whose bench base cannot reproduce the fold's own us/call is not priced at all.
        repro = entry["bench_reproduces_fold"]
        entry["priceable"] = bool(0.85 <= repro <= 1.20)
        entry["delta_s"] = (round(grp["calls"] * grp["us_per_call"]
                                  * (1.0 - rows[best]["ratio_to_bw1"]) / 1e6, 5)
                            if entry["priceable"] else 0.0)
        entry["band_s_if_priceable"] = round(grp["calls"] * grp["us_per_call"]
                                             * (1.0 - rows[best]["ratio_to_bw1"]) / 1e6, 5)
        res["groups"].append(entry)
        print("  %-20s kt=%-3d fold %8.3f us | bench bw1 %8.3f us (%.3fx of fold) | best bw=%d "
              "%8.3f us %.4fx | op-level delta %.4f s"
              % (",".join(str(v) for v in grp["shape"]), kt, grp["us_per_call"], base,
                 base / grp["us_per_call"], best, rows[best]["us_per_call"],
                 rows[best]["ratio_to_bw1"], entry["delta_s"]), flush=True)
        for bw in live:
            v = rows[bw]
            print("      bw=%-3d %9.3f us  %.4fx  spread %5.2f%%  equal=%s max_abs=%.3g"
                  % (bw, v["us_per_call"], v["ratio_to_bw1"], v["spread_pct"],
                     (v["values"] or {}).get("equal"), (v["values"] or {}).get("max_abs", 0)),
                  flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        if bias is not None:
            ttnn.deallocate(bias)

    res["env"]["clk_end"] = read_aiclk(a.node)
    res["env"]["t_close"] = time.time()
    tot = sum(x.get("delta_s", 0) for x in res["groups"])
    res["op_level_total_delta_s"] = round(tot, 5)
    out = HERE / ("bwladder_%s.json" % a.tag)
    out.write_text(json.dumps(res, indent=1))
    print("\nOP-LEVEL TOTAL over screened groups: %.4f s (each priced at its OWN shape's measured "
          "rate)" % tot)
    print("wrote", out, "clock at close", res["env"]["clk_end"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
