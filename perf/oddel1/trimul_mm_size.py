#!/usr/bin/env python3
"""Is the 640 -> 768 aa TriangleMultiplication cliff the triangle matmul's program config?

`_triangle_mul_program_config` sets per_core_M = ceil(Nt/10), per_core_N = ceil(Nt/13) on the
13x10 grid, so the padded work per core is per_core_M * per_core_N * Nt tile-MACs:

    Nt   20 (640 aa)   21 (672)   22 (704)   24 (768)   32 (1024)
    M,N  2,2           3,2        3,2        3,2        4,3
    work 80            126        132        144        384

Normalised by (Nt/16)^3 that is 41.0 / 55.7 / 55.6 / 42.7 / 48.0, i.e. the config predicts a
1.36x efficiency step between Nt=20 and Nt=21 and then a RECOVERY by Nt=24. The measured TriMul
efficiency curve steps between 640 and 768 and does NOT recover, so this arm exists to price how
much of the cliff the matmul can carry at all. `in0_block_w` also drops 10 -> 8 across the step.

Timed at OpenDDE's own chunk shape: [1, group*chunk, N, N] @ the same, bf16, DRAM, the trunk's
compute kernel config. 672 and 704 are the off-lattice rungs that separate "the config steps" from
"something else at 768"; per `sizes-recheck-opendde` lesson 2 a factor-of-2 ladder cannot localise
a defect that moves inside a doubling.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 2, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,640,672,704,768,1024")
    ap.add_argument("--channels", type=int, default=192)   # group 6 * chunk 32, opendde trunk
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "grid": [g.x, g.y], "channels": a.channels, "rows": []}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)

    for N in [int(s) for s in a.sizes.split(",")]:
        Nt = -(-N // 32)
        pc = T._triangle_mul_program_config(Nt)
        shp = [1, a.channels, N, N]
        aa = ttnn.from_torch(torch.zeros(shp, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        bb = ttnn.from_torch(torch.zeros(shp, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        def run():
            return ttnn.matmul(aa, bb, compute_kernel_config=ckc,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc,
                               dtype=ttnn.bfloat16)
        ts = []
        try:
            for _ in range(WARM):
                ttnn.deallocate(run())
            ttnn.synchronize_device(dev)
            for _ in range(REPS):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = run()
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
            ms = round(st.median(ts), 4)
            err = None
        except Exception as exc:                                       # noqa: BLE001
            ms, err = None, str(exc).strip().split("\n")[-1][:160]
        work = pc.per_core_M * pc.per_core_N * Nt
        row = {"N": N, "Nt": Nt, "per_core_M": pc.per_core_M, "per_core_N": pc.per_core_N,
               "in0_block_w": pc.in0_block_w, "padded_work_per_core": work,
               "work_per_512cube": round(work / (Nt / 16) ** 3, 3),
               "ms": ms, "ms_per_512cube": None if ms is None else round(ms / (N / 512) ** 3, 4),
               "error": err}
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
        ttnn.deallocate(aa); ttnn.deallocate(bb)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
