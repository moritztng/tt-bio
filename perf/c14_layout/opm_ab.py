#!/usr/bin/env python3
"""Op-level A/B: can OuterProductMean drop its 537 MB permute by contracting dim -2 in the matmul.

Site `tt_bio/tenstorrent.py:10307` is the second-largest permute in a 512 aa fold by bytes --
16 calls of `permute([512,1024,512], (0,2,1))`, 17.18 GB, 22.7 % of the whole Transpose+Permute
class -- and it exists for exactly one reason: `ttnn.linear` contracts the LAST dim, so the
(C*D)=1024 axis has to be moved there first. Nothing downstream wants that LAYOUT, it wants the
CONTRACTION. `ttnn.matmul(transpose_a=True)` contracts dim -2 directly, so the permute is a
candidate for deletion rather than acceleration.

WHAT `opm_isolate.py` FOUND FIRST, and why this file exists in this shape: scored against a host
float64 reference at [64,1024,512] x [1024,128] bf16,

    permute + linear, core_grid            rel_max_err 0.002246   OK
    permute + linear, no core_grid         rel_max_err 0.002246   OK
    transpose_a,      no core_grid         rel_max_err 0.004112   OK
    transpose_a,      core_grid            rel_max_err 1.226881   WRONG

`ttnn.matmul(transpose_a=True, core_grid=...)` returns a silently WRONG result -- no exception,
deviation larger than the signal. So arm B here runs transpose_a WITHOUT core_grid, and arm A2
(the shipped sequence also without core_grid) is carried so the grid is not doing the work in the
comparison: memory `roof-cube-control-kernel-config-must-match-arm` is the standing reason a
control whose kernel config differs from its arm is not a control.

This is an OP-LEVEL number. The C14 standing rule holds: nothing at op level in this campaign has
transferred to a fold -- four transferred at 25x-to-infinite error and two flipped sign. A win
here licenses BUILDING the lever and nothing else. Only an interleaved, benchlocked fold A/B with
its own A/A floor enters the ladder. No second in this file is bookable.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "perf/c10_bare_baseline")]

import torch                                                                   # noqa: E402
import ttnn                                                                    # noqa: E402
from force_aiclk import FORCE_AICLK, smc                                       # noqa: E402

TARGET_MHZ = 1350


def clock_stats(path, t0, t1):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    during = [r for r in rows if r["read_start_ns"] >= t0 and r["read_end_ns"] <= t1]
    mhz = [r["MHz"] for r in during if r.get("MHz")]
    if not mhz:
        return {"samples": 0, "qualified": False, "why": "no sample inside the timed interval"}
    centers = [(r["read_start_ns"] + r["read_end_ns"]) // 2 for r in during if r.get("MHz")]
    pts = [t0] + centers + [t1]
    gap = max(b - a for a, b in zip(pts, pts[1:]))
    return {"samples": len(mhz), "min_MHz": min(mhz), "max_MHz": max(mhz),
            "max_gap_ms": round(gap / 1e6, 2),
            "span_fraction": round((centers[-1] - centers[0]) / (t1 - t0), 4),
            "qualified": min(mhz) >= 1200 and gap <= 20_000_000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cd", type=int, default=1024)
    ap.add_argument("--j", type=int, default=512)
    ap.add_argument("--cout", type=int, default=128)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import tt_bio.tenstorrent as T
    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    # The SHIPPED config, read off tt_bio/tenstorrent.py:10844 (TorchWrapper). Every arm gets it.
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    grid = T.CORE_GRID_MAIN

    R = {"host": socket.gethostname(), "node": a.node, "pid": os.getpid(),
         "site": "tt_bio/tenstorrent.py:10307 (OuterProductMean output projection)",
         "shape_z": [a.rows, a.cd, a.j], "shape_w": [a.cd, a.cout],
         "ttnn": ttnn.__file__, "started_utc": time.time(),
         "note": "OP-LEVEL ONLY. Does not transfer to a fold. Not bookable.",
         "arms": {}, "equiv": {}, "errors": []}

    def save():
        (a.out / "opm_ab.json").write_text(json.dumps(R, indent=1, default=str))

    torch.manual_seed(0)
    zh = torch.randn(a.rows, a.cd, a.j) * 0.05
    wh = torch.randn(a.cd, a.cout) * 0.05
    bh = torch.randn(1, a.cout) * 0.05

    def dev_t(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

    z, w, b = dev_t(zh), dev_t(wh), dev_t(bh)

    def arm_a():
        p = ttnn.permute(z, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.linear(p, w, bias=b, compute_kernel_config=ckc, core_grid=grid)
        ttnn.deallocate(p)
        return o

    def arm_a2():
        p = ttnn.permute(z, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.linear(p, w, bias=b, compute_kernel_config=ckc)
        ttnn.deallocate(p)
        return o

    def arm_b():
        o = ttnn.matmul(z, w, transpose_a=True, compute_kernel_config=ckc)
        return ttnn.add_(o, b)

    ARMS = {"A": arm_a, "A2": arm_a2, "B": arm_b}

    # ---- equivalence at the production shape, device vs device ---------------------------
    outs = {}
    for tag, fn in ARMS.items():
        o = fn()
        ttnn.synchronize_device(dev)
        outs[tag] = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
    ref = outs["A"]
    den = ref.abs().max().item()
    for tag in ("A2", "B"):
        R["equiv"][tag + "_vs_A"] = {
            "bit_exact": bool(torch.equal(ref, outs[tag])),
            "max_abs": (ref - outs[tag]).abs().max().item(),
            "rel_max": (ref - outs[tag]).abs().max().item() / den}
    R["equiv"]["A_absmax"] = den
    R["equiv"]["out_shape"] = list(ref.shape)
    save()
    print(json.dumps(R["equiv"], indent=1), flush=True)

    # ---- interleaved timing, clock sampled THROUGH the region ---------------------------
    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    clk = None
    try:
        R["force_response"] = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
        time.sleep(0.2)
        budget = 40 + a.reps * 12
        clk = subprocess.Popen(
            [sys.executable, str(HERE / "clock_sampler.py"),
             str(a.out / "clock.jsonl"), str(a.node), str(budget)],
            stdout=subprocess.DEVNULL, stderr=open(str(a.out / "sampler.err"), "w"))
        time.sleep(0.5)
        for fn in ARMS.values():                                  # warm every arm, discard
            for _ in range(2):
                ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        t_start = time.monotonic_ns()
        times = {k: [] for k in ARMS}
        order = list(ARMS)
        for i in range(a.reps):
            seq = order if i % 2 == 0 else list(reversed(order))
            for tag in seq:
                ttnn.synchronize_device(dev)
                t0 = time.monotonic_ns()
                o = ARMS[tag]()
                ttnn.synchronize_device(dev)
                times[tag].append((time.monotonic_ns() - t0) / 1e6)
                ttnn.deallocate(o)
        t_end = time.monotonic_ns()
        for tag, v in times.items():
            R["arms"][tag] = {"n": len(v), "median_ms": round(st.median(v), 4),
                              "min_ms": round(min(v), 4), "max_ms": round(max(v), 4),
                              "all_ms": [round(x, 4) for x in v]}
        save()
        try:
            R["clock"] = clock_stats(a.out / "clock.jsonl", t_start, t_end)
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("clock sampling failed: %r -- NO SECOND HERE IS VALID" % (e,))
            R["clock"] = {"qualified": False, "error": repr(e)}
        ma = R["arms"]["A"]["median_ms"]
        R["ratios_vs_A"] = {t: round(ma / R["arms"][t]["median_ms"], 4) for t in ARMS}
        R["completed"] = True
    finally:
        try:
            R["release_response"] = list(smc(fd, FORCE_AICLK, 0))
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("clock release failed: %r" % (e,))
        os.close(fd)
        if clk is not None:
            clk.wait(timeout=120)
        save()
    print(json.dumps({k: R[k] for k in ("arms", "ratios_vs_A", "clock") if k in R},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
