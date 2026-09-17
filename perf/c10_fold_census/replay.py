#!/usr/bin/env python3
"""Measure, on qb2 card 0 at a pinned and during-sampled 1350 MHz, what the 512 aa fold's own
recorded shapes can actually do, and what the compute and bandwidth roofs are in the same session.

Every arm is a launch key the fold really issues, taken from `perf/roof_launch/fold_shapes.json`
with its own call weight, so a rate measured here converts to fold seconds through the fold's own
call census and not through a phase total divided by an op count.

Discipline, in the order it matters:
  * The dense cube is measured in THIS session, twice, once before the arms and once after. A
    cube taken in another session at another clock is not a reference for these rows.
  * Both counters pass the known-answer control (8192^3 -> exactly 1,099,511,627,776 FLOPs and
    exactly 402,653,184 bytes) before any timing starts. This campaign has had three byte-counter
    defects; the control is two lines and it is not optional.
  * The clock is requested through ARC FORCE_AICLK and sampled at about 1 kHz DURING every timed
    interval by a separate process. An interval whose during-samples are not min == max == 1350,
    or which has a gap above 10 ms or a read error, is an ARTIFACT and is dropped, not reported.
  * Where two configurations of the same key are plausible the FASTER one is kept, so each rate
    is an upper bound on what the shape can do and the fold-seconds it implies are a LOWER bound.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import census as C                                                            # noqa: E402
import control                                                                # noqa: E402
from force_aiclk import FORCE_AICLK, smc                                       # noqa: E402

TARGET_MHZ = 1350
LOOP_TARGET_S = 0.030          # aim each timed loop at 30 ms: long enough to amortise a sync
MAX_REPS = 64
LIVE_OUTPUTS = 4               # bound the live footprint while keeping the enqueue depth


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def build_roofs(ttnn, torch, dev, kc):
    """The two roofs, measured here, in this session, at this clock.

    cube8192 is the dense bf16 HiFi4 compute roof and carries the exact known-answer counts.
    dram8192 is a starved elementwise add over two 8192x8192 operands: no reuse is possible, so
    its 402,653,184 bytes (the same count as the cube's minimum) measure the DRAM roof.
    """
    def t(shape):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def cube(n):
        def make():
            a, b = t([n, n]), t([n, n])
            return [a, b], (lambda: ttnn.matmul(a, b, compute_kernel_config=kc,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG)), "out"
        return {"key": "roof.cube%d" % n, "arm": "roof_cube", "out": [n, n], "K": n, "calls": 0.0,
                "flops_per_call": C.matmul_flops([n, n], n),
                "min_bytes_per_call": 3 * C.tensor_bytes([n, n]), "make": make}

    def dram(n):
        def make():
            a, b = t([n, n]), t([n, n])
            return [a, b], (lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)), "out"
        return {"key": "roof.dram_add%d" % n, "arm": "roof_dram", "out": [n, n], "K": None,
                "calls": 0.0, "flops_per_call": 0,
                "min_bytes_per_call": 3 * C.tensor_bytes([n, n]), "make": make}

    return [cube(8192), dram(8192), cube(4096)]


def attach(ttnn, torch, dev, kc, grid, spec):
    """Give a census arm spec a `make()` that allocates its operands and returns its callable."""
    def t(shape, ones=False):
        x = (torch.ones(*shape, dtype=torch.bfloat16) if ones
             else torch.randn(*shape, dtype=torch.bfloat16))
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    arm, out, k = spec["arm"], spec["out"], spec["K"]
    shapes = {n: s for n, s in spec["operands"]}
    DRAMC = ttnn.DRAM_MEMORY_CONFIG

    if arm == "linear":
        def make(g=None):
            a, w = t(shapes["a"]), t(shapes["w"])
            kw = {"core_grid": g} if g is not None else {}
            return [a, w], (lambda: ttnn.linear(a, w, compute_kernel_config=kc,
                                                memory_config=DRAMC, dtype=ttnn.bfloat16,
                                                **kw)), "out"
    elif arm == "matmul":
        def make(g=None):
            a, b = t(shapes["a"]), t(shapes["b"])
            kw = {"core_grid": g} if g is not None else {}
            return [a, b], (lambda: ttnn.matmul(a, b, compute_kernel_config=kc,
                                                memory_config=DRAMC, **kw)), "out"
    elif arm in C.INPLACE_ARMS:
        op = ttnn.multiply_ if arm == "multiply_" else ttnn.add_

        def make(g=None):
            # ones, not randn: an in-place op applied thousands of times to its own destination
            # would otherwise saturate bf16 and the numbers stop meaning anything. Timing on
            # Tensix does not depend on the values, and a saturated operand is not a measurement
            # anyone should have to defend.
            a, b = t(shapes["a"], ones=True), t(shapes["b"], ones=True)
            return [a, b], (lambda: op(a, b)), "inplace"
    elif arm in C.ELTWISE_ARMS:
        op = ttnn.multiply if arm == "multiply" else ttnn.add

        def make(g=None):
            a, b = t(shapes["a"]), t(shapes["b"])
            return [a, b], (lambda: op(a, b, memory_config=DRAMC)), "out"
    elif arm == "layer_norm":
        def make(g=None):
            a = t(shapes["a"])
            return [a], (lambda: ttnn.layer_norm(a, compute_kernel_config=kc,
                                                 memory_config=DRAMC)), "out"
    elif arm == "layer_norm_w":
        def make(g=None):
            a, w, b = t(shapes["a"]), t(shapes["w"]), t(shapes["b"])
            return [a, w, b], (lambda: ttnn.layer_norm(a, weight=w, bias=b,
                                                       compute_kernel_config=kc,
                                                       memory_config=DRAMC)), "out"
    else:
        return None
    spec = dict(spec)
    spec["make"] = make
    return spec


def time_arm(ttnn, dev, spec, blocks, grid=None, log=print):
    """Allocate, warm, time `blocks` loops, free. Returns the best loop and its interval marks."""
    try:
        keep, fn, mode = spec["make"](grid) if grid is not None else spec["make"]()
    except TypeError:
        keep, fn, mode = spec["make"]()
    live, best, marks = [], None, []
    try:
        t0 = time.perf_counter()
        for _ in range(2):
            r = fn()
            if mode == "out":
                ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        warm = (time.perf_counter() - t0) / 2
        reps = max(2, min(MAX_REPS, int(LOOP_TARGET_S / warm) if warm > 0 else MAX_REPS))
        for _ in range(blocks):
            outs = []
            s_ns = time.monotonic_ns()
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
                if mode == "out":
                    outs.append(r)
                    if len(outs) > LIVE_OUTPUTS:
                        ttnn.deallocate(outs.pop(0))
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / reps
            e_ns = time.monotonic_ns()
            for o in outs:
                ttnn.deallocate(o)
            marks.append({"start_monotonic_ns": s_ns, "end_monotonic_ns": e_ns,
                          "reps": reps, "s_per_call": dt})
            best = dt if best is None else min(best, dt)
    finally:
        for x in keep:
            try:
                ttnn.deallocate(x)
            except Exception:                                                  # noqa: BLE001
                pass
    return {"s_per_call": best, "reps": reps, "marks": marks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--grid-control", type=int, default=6,
                    help="how many top-FLOP matmul keys also run on the fold's CORE_GRID_MAIN")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)

    cube = C.cube_control()
    if not (cube["flops_pass"] and cube["bytes_pass"]):
        raise RuntimeError("known-answer counter control failed: %s" % cube)

    shapes, op_census = C.load(ROOT / "perf")
    identity = C.byte_identity(op_census)
    specs, refused = C.build_arms(shapes)

    pre = control.snapshot()
    control.validate_snapshot(pre)
    preflight = {
        "host": os.uname().nodename, "node": a.node, "pid": os.getpid(),
        "git_rev": git("rev-parse", "HEAD").strip(),
        "git_status": git("status", "--porcelain").strip().splitlines(),
        "imports": json.loads((HERE / "imports.json").read_text()),
        "inputs": [control.digest(ROOT / "perf/roof_launch/fold_shapes.json"),
                   control.digest(ROOT / "perf/roof_launch/op_census_512.json")],
        "cube_control": cube, "byte_identity": {k: v for k, v in identity.items()
                                                if k != "rows"},
        "arms": len(specs), "refused": refused, "pre_snapshot": pre,
    }
    for entry in preflight["imports"]:
        got = control.digest(HERE / entry["file"])["sha256"]
        if got != entry["sha256"]:
            raise RuntimeError("pinned helper %s drifted: %s" % (entry["file"], got))
    (out / "preflight.json").write_text(json.dumps(preflight, indent=1) + "\n")
    print("known-answer control PASS: %d FLOPs, %d bytes" % (cube["flops"], cube["bytes"]),
          flush=True)
    print("byte identity: %d shapes, median %.6f, %d real holes worth %.3f GB (%.2f %%)"
          % (identity["shapes_checked"], identity["median_ratio"], len(identity["real_holes"]),
             identity["real_hole_bytes"] / 1e9, identity["real_hole_pct_of_recorded"]), flush=True)

    result = {"preflight": preflight, "target_MHz": TARGET_MHZ, "blocks": a.blocks, "rows": [],
              "errors": {}}
    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    sampler = dev = None
    released = False

    def release():
        nonlocal released
        if released:
            return
        released = True
        try:
            result["release_response"] = list(smc(fd, FORCE_AICLK, 0))
        except Exception as e:                                                # noqa: BLE001
            result["release_error"] = repr(e)

    def interrupted(sig, frame):
        raise RuntimeError("signal %d: releasing clock and device" % sig)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, interrupted)

    try:
        result["force_response"] = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
        if result["force_response"][0] != 0:
            raise RuntimeError("FORCE_AICLK(%d) refused: %s" % (TARGET_MHZ,
                                                                result["force_response"]))
        sampler = subprocess.Popen([sys.executable, str(HERE / "control.py"),
                                    str(out / "clock.jsonl"), str(os.getpid())],
                                   stdin=subprocess.PIPE,
                                   stdout=(out / "sampler.log").open("w"),
                                   stderr=subprocess.STDOUT)
        time.sleep(1.0)

        import torch                                                          # noqa: PLC0415
        import ttnn                                                           # noqa: PLC0415
        from tt_bio import tenstorrent as TT                                   # noqa: PLC0415
        from tt_bio import af2                                                 # noqa: PLC0415

        dev = TT.get_device()
        control.validate_snapshot(control.snapshot(opened=True), opened=True)
        kc = af2.compute_kernel_config()
        grid = TT.CORE_GRID_MAIN
        cc = dev.compute_with_storage_grid_size()
        result["device"] = {"arch": str(dev.arch()), "grid": [cc.x, cc.y],
                            "core_grid_main": [grid.x, grid.y],
                            "cores_core_grid_main": grid.x * grid.y,
                            "kernel_config": str(kc),
                            "ttnn": getattr(ttnn, "__version__", "unknown")}
        print("device %s grid %dx%d, CORE_GRID_MAIN %dx%d"
              % (result["device"]["arch"], cc.x, cc.y, grid.x, grid.y), flush=True)

        roofs = build_roofs(ttnn, torch, dev, kc)
        live = [attach(ttnn, torch, dev, kc, grid, s) for s in specs]
        live = [s for s in live if s]
        if a.only:
            want = set(a.only.split(","))
            live = [s for s in live if s["key"] in want]
        mm = [s for s in live if s["arm"] in C.MATMUL_ARMS]
        mm.sort(key=lambda s: -s["flops_per_call"] * s["calls"])
        grid_keys = {s["key"] for s in mm[:a.grid_control]}

        order = [roofs[0], roofs[1], roofs[2]] + live + [dict(roofs[0], key="roof.cube8192_end")]
        for i, spec in enumerate(order):
            variants = [(spec["key"], None)]
            if spec["key"] in grid_keys:
                variants.append((spec["key"] + "@grid110", grid))
            for label, g in variants:
                try:
                    r = time_arm(ttnn, dev, spec, a.blocks, grid=g)
                except Exception as e:                                        # noqa: BLE001
                    result["errors"][label] = "%s: %s" % (type(e).__name__,
                                                          str(e).splitlines()[0][:200])
                    print("DROP %-44s %s" % (label, result["errors"][label]), flush=True)
                    try:
                        ttnn.synchronize_device(dev)
                    except Exception:                                         # noqa: BLE001
                        pass
                    continue
                row = {"label": label, "key": spec["key"], "arm": spec["arm"],
                       "out": spec["out"], "K": spec["K"], "calls": spec["calls"],
                       "grid": [g.x, g.y] if g is not None else None,
                       "flops_per_call": spec["flops_per_call"],
                       "min_bytes_per_call": spec["min_bytes_per_call"], **r}
                row["TFLOPs"] = (spec["flops_per_call"] / r["s_per_call"] / 1e12
                                 if spec["flops_per_call"] else None)
                row["GBs"] = spec["min_bytes_per_call"] / r["s_per_call"] / 1e9
                result["rows"].append(row)
                print("%-44s %9.4f ms  %8.2f TFLOP/s %8.1f GB/s  reps=%d"
                      % (label, r["s_per_call"] * 1e3, row["TFLOPs"] or 0, row["GBs"],
                         r["reps"]), flush=True)
            if i % 8 == 0:
                (out / "replay.partial.json").write_text(json.dumps(result, indent=1) + "\n")
    finally:
        release()
        if dev is not None:
            try:
                import tt_bio.tenstorrent as TT2                              # noqa: PLC0415
                TT2.cleanup()
            except Exception as e:                                            # noqa: BLE001
                result["cleanup_error"] = repr(e)
        if sampler is not None:
            try:
                sampler.stdin.write(b"stop\n")
                sampler.stdin.flush()
                sampler.stdin.close()
            except Exception:                                                 # noqa: BLE001
                pass
            sampler.wait(timeout=30)
        os.close(fd)
        result["post_snapshot"] = control.snapshot()
        samples = [json.loads(x) for x in (out / "clock.jsonl").read_text().splitlines() if x]
        holders = [json.loads(x) for x in (out / "holders.jsonl").read_text().splitlines()
                   if x] if (out / "holders.jsonl").exists() else []
        for row in result["rows"]:
            row["clock"] = [control.coverage(samples, m) for m in row["marks"]]
            row["clock_pass"] = all(c["pass"] for c in row["clock"])
            ok = [m["s_per_call"] for m, c in zip(row["marks"], row["clock"]) if c["pass"]]
            row["s_per_call_qualified"] = min(ok) if ok else None
            row["blocks_qualified"] = len(ok)
        result["clock_samples"] = len(samples)
        result["clock_min_MHz"] = min((s["MHz"] for s in samples if "MHz" in s), default=None)
        result["clock_max_MHz"] = max((s["MHz"] for s in samples if "MHz" in s), default=None)
        result["clock_read_errors"] = sum(1 for s in samples if "error" in s)
        result["foreign_holders"] = [h for h in holders
                                     if any(x["pid"] != os.getpid() for x in h.get("holders", []))]
        (out / "replay.json").write_text(json.dumps(result, indent=1) + "\n")
        for name in ("clock.jsonl", "holders.jsonl"):
            p = out / name
            if p.exists():
                subprocess.run(["gzip", "-n", "-f", str(p)], check=False)
        rp = out / "replay.partial.json"
        if rp.exists():
            rp.unlink()
    if result.get("release_response", [None])[0] != 0:
        raise RuntimeError("clock release not confirmed: %s" % result.get("release_response"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
