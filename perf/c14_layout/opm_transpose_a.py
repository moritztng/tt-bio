#!/usr/bin/env python3
"""Can OuterProductMean drop its 537 MB permute by letting the matmul contract dim -2?

Site `tenstorrent.py:10307` is the second-largest permute in the fold by bytes: 16 calls of
`permute([512,1024,512], (0,2,1))`, 17.18 GB = 22.7 % of the whole Transpose+Permute class,
and it exists for exactly one reason -- `ttnn.linear` contracts the LAST dim, so the
(C*D) axis has to be moved there. `ttnn.matmul` takes `transpose_a`, which contracts dim -2
directly, so the permute is a candidate for outright deletion rather than acceleration.

THE RISK THIS RIG EXISTS TO CATCH is the one that killed `c14-radical`s lead candidate: a lever
that is already taken, or that only moves the cost. If `transpose_a=True` is implemented as an
explicit transpose inside the matmul, the bytes do not go away and arm B is not faster. That is
not answerable from source; the tt-metal matmul folds a transpose into the in0 reader for some
program configs and not others.

Arms, interleaved A B B A so compile and warm-up bias cannot land on one of them:
  A  permute(z,(0,2,1)) -> linear(w)        the shipped sequence
  B  matmul(z, w, transpose_a=True)         the same contraction, no permute
  C  NEGATIVE CONTROL: arm B with the weight rows reversed. It must NOT match arm A. Without it
     a `transpose_a` that silently contracted the wrong axis would read as a free win.

Equivalence is checked BOTH ways round, because the two failure modes are different:
  * arm B vs arm A on device -- is the reindex the same reindex
  * arm A vs a float64 HOST reference at a small shape -- is the shipped sequence itself what we
    think it is. A device-vs-device check cannot see a shared misunderstanding.

This is an OP-LEVEL number. The campaign rule stands: nothing at op level in C14 has transferred
to a fold (four transferred at 25x-to-infinite error, two flipped sign), so a win here licenses
building the lever and nothing else. No second in this file may be entered in any book.
"""
from __future__ import annotations

import argparse, json, os, socket, statistics as st, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "perf/c10_bare_baseline")]

import torch                                                                   # noqa: E402
import ttnn                                                                    # noqa: E402
from force_aiclk import FORCE_AICLK, smc                                       # noqa: E402

TARGET_MHZ = 1350


def sample_clock(node, path, stop_after):
    """Sample tt_aiclk at ~1 kHz in a subprocess THROUGH the timed region."""
    code = (
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "p = Path(sys.argv[1]); node = sys.argv[2]; dl = time.monotonic() + float(sys.argv[3])\n"
        "src = Path(/sys/class/tenstorrent/tenstorrent!%s/tt_aiclk % node)\n"
        "with p.open(w) as f:\n"
        "    while time.monotonic() < dl:\n"
        "        a = time.monotonic_ns()\n"
        "        try: m = int(src.read_text())\n"
        "        except Exception as e: m = None\n"
        "        f.write(json.dumps({read_start_ns: a, MHz: m,\n"
        "                            read_end_ns: time.monotonic_ns()}) + chr(10)); f.flush()\n")
    return subprocess.Popen([sys.executable, "-c", code, str(path), str(node), str(stop_after)],
                            stdout=subprocess.DEVNULL,
                            stderr=open(str(path) + ".err", "w"))


def clock_stats(path, t0, t1):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    during = [r for r in rows if r["read_start_ns"] >= t0 and r["read_end_ns"] <= t1]
    mhz = [r["MHz"] for r in during if r.get("MHz")]
    if not mhz:
        return {"samples": 0, "min_MHz": None, "max_MHz": None, "qualified": False}
    centers = [(r["read_start_ns"] + r["read_end_ns"]) // 2 for r in during if r.get("MHz")]
    pts = [t0] + centers + [t1]
    gap = max(b - a for a, b in zip(pts, pts[1:]))
    return {"samples": len(mhz), "min_MHz": min(mhz), "max_MHz": max(mhz),
            "max_gap_ms": round(gap / 1e6, 2),
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
    # The SHIPPED config, read off tt_bio/tenstorrent.py:10844 (TorchWrapper). Both arms get it.
    # Memory roof-cube-control-kernel-config-must-match-arm: the same cube read 1.40x apart on
    # fp32_dest_acc_en/packer_l1_acc alone, so an unmatched control is not a control.
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    grid = T.CORE_GRID_MAIN

    R = {"host": socket.gethostname(), "node": a.node, "pid": os.getpid(),
         "shape_z": [a.rows, a.cd, a.j], "shape_w": [a.cd, a.cout],
         "ttnn": ttnn.__file__, "started_utc": time.time(),
         "site": "tt_bio/tenstorrent.py:10307 (OuterProductMean output projection)",
         "note": "OP-LEVEL ONLY. Does not transfer to a fold. Not bookable.",
         "arms": {}, "equiv": {}, "errors": []}

    def save():
        (a.out / "opm_transpose_a.json").write_text(json.dumps(R, indent=1, default=str))

    # ---- known-answer host control at a small shape, float64 reference -------------------
    r0, cd0, j0, co0 = 2, 64, 32, 16
    zh = torch.randn(r0, cd0, j0, dtype=torch.float32)
    wh = torch.randn(cd0, co0, dtype=torch.float32)
    ref = torch.einsum("rcj,co->rjo", zh.double(), wh.double())
    zt = ttnn.from_torch(zh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wt = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pa = ttnn.permute(zt, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    oa = ttnn.matmul(pa, wt, compute_kernel_config=ckc)
    ob = ttnn.matmul(zt, wt, transpose_a=True, compute_kernel_config=ckc)
    ha, hb = ttnn.to_torch(oa).float(), ttnn.to_torch(ob).float()
    wrev = ttnn.from_torch(torch.flip(wh, dims=[0]), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    oc = ttnn.matmul(zt, wrev, transpose_a=True, compute_kernel_config=ckc)
    hc = ttnn.to_torch(oc).float()
    den = ref.abs().max().item()
    R["equiv"]["small_shape"] = [r0, cd0, j0, co0]
    R["equiv"]["A_vs_float64_rel"] = (ha.double() - ref).abs().max().item() / den
    R["equiv"]["B_vs_float64_rel"] = (hb.double() - ref).abs().max().item() / den
    R["equiv"]["A_vs_B_bit_exact"] = bool(torch.equal(ha, hb))
    R["equiv"]["A_vs_B_max_abs"] = (ha - hb).abs().max().item()
    R["equiv"]["negctrl_C_differs"] = not bool(torch.equal(ha, hc))
    R["equiv"]["negctrl_C_max_abs"] = (ha - hc).abs().max().item()
    for t in (zt, wt, wrev, pa, oa, ob, oc):
        ttnn.deallocate(t)
    save()
    print(json.dumps(R["equiv"], indent=1), flush=True)
    if not R["equiv"]["negctrl_C_differs"]:
        R["errors"].append("NEGATIVE CONTROL FAILED: weight-reversed arm matched. stop.")
        save()
        return 2
    if R["equiv"]["B_vs_float64_rel"] > 4 * max(R["equiv"]["A_vs_float64_rel"], 1e-6):
        R["errors"].append("arm B is not the same contraction as arm A. stop.")
        save()
        return 3

    # ---- full shape: equivalence, then interleaved timing -------------------------------
    zh = torch.randn(a.rows, a.cd, a.j, dtype=torch.float32) * 0.05
    wh = torch.randn(a.cd, a.cout, dtype=torch.float32) * 0.05
    z = ttnn.from_torch(zh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias = ttnn.from_torch(torch.randn(1, a.cout, dtype=torch.float32) * 0.05,
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def arm_a():
        p = ttnn.permute(z, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.linear(p, w, bias=bias, compute_kernel_config=ckc, core_grid=grid)
        ttnn.deallocate(p)
        return o

    def arm_b():
        o = ttnn.matmul(z, w, transpose_a=True, compute_kernel_config=ckc, core_grid=grid)
        return ttnn.add_(o, bias)

    oa, ob = arm_a(), arm_b()
    ttnn.synchronize_device(dev)
    ha, hb = ttnn.to_torch(oa).float(), ttnn.to_torch(ob).float()
    R["equiv"]["full_out_shape"] = list(ha.shape)
    R["equiv"]["full_A_vs_B_bit_exact"] = bool(torch.equal(ha, hb))
    R["equiv"]["full_A_vs_B_max_abs"] = (ha - hb).abs().max().item()
    R["equiv"]["full_A_absmax"] = ha.abs().max().item()
    ttnn.deallocate(oa); ttnn.deallocate(ob)
    save()

    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    clk = None
    try:
        R["force_response"] = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
        time.sleep(0.2)
        budget = 30 + a.reps * 8
        clk = sample_clock(a.node, a.out / "clock.jsonl", budget)
        time.sleep(0.3)
        for _ in range(2):                                     # warm both, discard
            ttnn.deallocate(arm_a()); ttnn.deallocate(arm_b())
        ttnn.synchronize_device(dev)
        t_start = time.monotonic_ns()
        times = {"A": [], "B": []}
        for i in range(a.reps):
            for tag in (("A", "B", "B", "A") if i % 2 == 0 else ("B", "A", "A", "B")):
                fn = arm_a if tag == "A" else arm_b
                ttnn.synchronize_device(dev)
                t0 = time.monotonic_ns()
                o = fn()
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
        except Exception as e:
            R["errors"].append("clock sampling failed: %r -- NO SECOND HERE IS VALID" % (e,))
            R["clock"] = {"qualified": False, "error": repr(e)}
        ma, mb = R["arms"]["A"]["median_ms"], R["arms"]["B"]["median_ms"]
        R["ratio_A_over_B"] = round(ma / mb, 4)
        R["B_over_A"] = round(mb / ma, 4)
        R["completed"] = True
    finally:
        try:
            R["release_response"] = list(smc(fd, FORCE_AICLK, 0))
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("clock release failed: %r" % (e,))
        os.close(fd)
        if clk is not None:
            clk.wait(timeout=60)
        save()
    print(json.dumps({k: R[k] for k in ("arms", "ratio_A_over_B", "clock") if k in R},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
