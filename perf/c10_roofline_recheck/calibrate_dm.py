"""Clocked DM calibration using the add/matmul controls from 4d80e855.

Run under the existing k10 Tracy build with --enable-sum-profiling.
These are instrument controls, not a model benchmark or a roof by assertion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types

from control import CLOCK_REF, ROOT, device_holders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if device_holders():
        raise RuntimeError("Quiet-board prerequisite failed")
    import torch
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import tt_bio.tenstorrent as T
    if not str(Path(ttnn.__file__).resolve()).startswith("/home/ttuser/tt-metal-k10/"):
        raise RuntimeError(f"Wrong ttnn: {ttnn.__file__}")
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    source = subprocess.check_output(["git", "show", CLOCK_REF], cwd=ROOT)
    clock = types.ModuleType("c10_clock")
    exec(compile(source, CLOCK_REF, "exec"), clock.__dict__)
    dev = T.get_device()
    sampler = None
    result = {
        "host": socket.gethostname(), "pid": os.getpid(), "ttnn": ttnn.__file__,
        "instrumentation_ref": "4d80e855e7fc75a436d6506359459349bb002999",
        "clock_source": CLOCK_REF, "clock_source_sha256": hashlib.sha256(source).hexdigest(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "flags": {k:v for k,v in os.environ.items() if k.startswith(("TT_", "PYTHONPATH", "LD_LIBRARY_PATH"))},
        "intervals": [], "reps": args.reps,
        "scope": "Instrument controls only; graph compulsory bytes omit physical rereads/spills.",
    }
    try:
        clock.engage("blackhole")
        if clock.status().get("nodes") != [0]:
            raise RuntimeError(f"Wrong clock node: {clock.status()}")
        sampler = subprocess.Popen([sys.executable, str(Path(__file__).with_name("control.py")),
                                    "--clock-worker", str(args.out/"clock.jsonl")],
                                   stdin=subprocess.PIPE, text=True)
        fence_tensor = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16),
                                     layout=ttnn.TILE_LAYOUT, device=dev)
        def fence():
            for _ in range(3):
                z = ttnn.exp(fence_tensor)
                ttnn.deallocate(z)
            ttnn.synchronize_device(dev)

        for case, n in (("stream_add", 8192), ("dense_matmul", 2048)):
            def make():
                return ttnn.from_torch(torch.ones(n, n, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x, y = make(), make()
            cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
            def call():
                if case == "stream_add":
                    return ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return ttnn.matmul(x, y, compute_kernel_config=cfg,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(3):
                z = call()
                ttnn.synchronize_device(dev)
                ttnn.deallocate(z)
            fence()
            start = time.monotonic_ns()
            for _ in range(args.reps):
                z = call()
                ttnn.deallocate(z)
            ttnn.synchronize_device(dev)
            end = time.monotonic_ns()
            fence()
            result["intervals"].append({
                "case":case, "n":n, "start_monotonic_ns":start, "end_monotonic_ns":end,
                "host_wall_ns":end-start, "reps":args.reps,
                "expected_compulsory_bytes_per_call":3*n*n*2,
                "expected_flops_per_call":n*n if case=="stream_add" else 2*n**3,
            })
            ttnn.deallocate(x)
            ttnn.deallocate(y)
        result["clock_hold"] = clock.status()
        result["loaded_shared_libraries"] = sorted({l.split()[-1] for l in Path("/proc/self/maps").read_text().splitlines() if "tt-metal" in l})
        sampler.communicate("stop\n", timeout=10)
        samples = [json.loads(l) for l in (args.out/"clock.jsonl").read_text().splitlines()]
        for interval in result["intervals"]:
            start, end = interval["start_monotonic_ns"], interval["end_monotonic_ns"]
            during = [s for s in samples if s.get("read_start_ns",0)>=start and s.get("read_end_ns",end+1)<=end]
            centers = [(s["read_start_ns"]+s["read_end_ns"])//2 for s in during]
            points = [start]+centers+[end]
            coverage = {
                "samples":len(during),
                "min_MHz":min((s["MHz"] for s in during), default=None),
                "max_MHz":max((s["MHz"] for s in during), default=None),
                "max_gap_ns":max(b-a for a,b in zip(points,points[1:])),
                "sample_span_fraction":(centers[-1]-centers[0])/(end-start) if centers else 0,
                "during":during,
            }
            coverage["pass"] = (len(during)>=3 and coverage["min_MHz"]==coverage["max_MHz"]==1350
                                and coverage["max_gap_ns"]<=10_000_000)
            interval["clock"] = coverage
        result["foreign_holders"] = [s for s in samples if s.get("foreign_holders")]
        result["clock_errors"] = [s for s in samples if s.get("error")]
        result["clock_pass"] = (all(i["clock"]["pass"] for i in result["intervals"])
                               and not result["foreign_holders"] and not result["clock_errors"])
        if not result["clock_pass"]:
            raise RuntimeError("Clock or quiet-board calibration failed")
        print("CAPTURE COMPLETE: runtime DM-zone validation still required", flush=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if sampler is not None and sampler.poll() is None:
            sampler.communicate("stop\n", timeout=10)
        (args.out/"capture.json").write_text(json.dumps(result,indent=2)+"\n")
        clock.release()
        T.cleanup()


if __name__ == "__main__":
    main()

