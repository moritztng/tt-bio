#!/usr/bin/env python3
"""bc4-bet-contract probe entry: `probe.py control OUT` times the chip control, `probe.py sweep ARGS...` runs
perf/spd/af2_trimul_bmm_sweep.py. Both open the card through tt-bio with the p300 mesh descriptor set first.

The control is one large square bf16 matmul in DRAM at the trunk's kernel config (HiFi4, fp32 dest, packer L1
acc) and at HiFi2: the rate this chip attains on a matmul that is not setup- or traffic-bound, so the trimul
contraction's per-call rates read against it.
"""
import json, runpy, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()


def control(out, n=4096, reps=20, inner=5):
    import torch
    import ttnn
    from tt_bio import tenstorrent as tt
    dev = tt.get_device()
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    torch.manual_seed(0)
    x, y = up(torch.randn(1, 1, n, n)), up(torch.randn(1, 1, n, n))
    rows = []
    for fid in ("HiFi4", "HiFi2"):
        ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                      math_approx_mode=False, fp32_dest_acc_en=True,
                                                      packer_l1_acc=True)
        fn = lambda: ttnn.matmul(x, y, compute_kernel_config=ckc, dtype=ttnn.bfloat16)  # noqa: E731
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(inner):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / inner)
        ms = 1e3 * statistics.median(ts)
        rows.append({"arm": f"square{n}_{fid}", "ms": ms, "ms_min": 1e3 * min(ts), "ms_max": 1e3 * max(ts),
                     "tflops": 2 * n ** 3 / (ms * 1e-3) / 1e12, "reps": reps, "inner": inner})
        print(rows[-1], flush=True)
    Path(out).write_text(json.dumps({"control": rows, "loadavg": open("/proc/loadavg").read().split()[:3]},
                                    indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "control":
        control(sys.argv[2])
    else:
        sys.argv = ["af2_trimul_bmm_sweep.py"] + sys.argv[2:]
        runpy.run_path(str(REPO / "perf/spd/af2_trimul_bmm_sweep.py"), run_name="__main__")
