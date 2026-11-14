#!/usr/bin/env python3
"""mol1-bh-prof: a matmul ceiling table on one p150a at Mol-1's pair-stack shapes (1024 tokens).

Per shape, ttnn.matmul on its own plan at bf16 HiFi4 (what the fold runs), HiFi2 and LoFi, with bfp8_b weights at
HiFi2 and LoFi, and, for OPM's contraction, Wormhole's tuned `opm_contract_config` plan built for this grid. fp32 dest
accumulation and packer L1 accumulation throughout. Median of --reps calls, one sync per call, AICLK of every card
sampled during the calls. Error is rel L2 against a float64 product on the first 256 output rows.

    TT_VISIBLE_DEVICES=<card> timeout 1500 python mm_ceiling.py out.json [--only opm]
"""
import argparse, glob, json, os, pathlib, statistics, sys, threading, time
import torch

ROOT = pathlib.Path(os.environ.get("TT_BIO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))

N = 1024
# name: (M, K, Nout, transpose_b, batch)  -- A [batch, M, K] x B [K, Nout] (or [Nout, K] with transpose_b)
SHAPES = {
    "opm contract (128-row block, depth 6144)": (128 * 32, 6144, N * 32, True, 1),
    "opm contract (whole, depth 6144)": (N * 32, 6144, N * 32, True, 1),
    "opm proj_o": (N * N, 1024, 448, False, 1),
    "pair linear c448 -> 448": (N * N, 448, 448, False, 1),
    "pair linear c320 -> 320": (N * N, 320, 320, False, 1),
    "pair transition c320 up (x4)": (N * N, 320, 1280, False, 1),
    "pair transition c320 down": (N * N, 1280, 320, False, 1),
    "triattn qkv c320": (N * N, 320, 960, False, 1),
    "trimul einsum, 32 channels": (N, N, N, False, 32),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--only", default="")
    p.add_argument("--reps", type=int, default=5)
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    grid = dev.compute_with_storage_grid_size()
    ckc = lambda f: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=f, fp32_dest_acc_en=True,
                                                           packer_l1_acc=True)
    F = ttnn.MathFidelity
    clk_paths = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append([int(open(f).read()) for f in clk_paths])
            time.sleep(0.1)
    threading.Thread(target=poll, daemon=True).start()
    g = torch.Generator().manual_seed(0)
    res = {"host": os.uname().nodename, "sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "grid": [grid.x, grid.y], "clk_paths": clk_paths, "shapes": {}}
    for name, (M, K, Nn, tb, bt) in SHAPES.items():
        if A.only and A.only not in name:
            continue
        a_t = torch.randn(bt, M, K, generator=g).to(torch.bfloat16)
        b_t = (torch.randn(bt if bt > 1 else 1, *((Nn, K) if tb else (K, Nn)), generator=g) * K ** -0.5).to(torch.bfloat16)
        ref = (a_t[:, :256].double() @ (b_t.double().transpose(-1, -2) if tb else b_t.double()))
        a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        b16 = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        b8 = ttnn.from_torch(b_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        arms = {"bf16 HiFi4 (fold)": (b16, F.HiFi4, None), "bf16 HiFi2": (b16, F.HiFi2, None),
                "bf16 LoFi": (b16, F.LoFi, None), "bfp8 w HiFi2": (b8, F.HiFi2, None), "bfp8 w LoFi": (b8, F.LoFi, None)}
        if name.startswith("opm contract"):
            pc = TT.opm_contract_config(M // 32, Nn // 32, K // 32, grid)
            if pc is not None:
                arms["bf16 HiFi4 wh-plan"] = (b16, F.HiFi4, pc)
                arms["bf16 HiFi2 wh-plan"] = (b16, F.HiFi2, pc)
        fl = 2.0 * bt * M * K * Nn
        row = {"M": M, "K": K, "N": Nn, "batch": bt, "tflop": fl / 1e12, "arms": {}}
        for arm, (b, fid, pc) in arms.items():
            kw = dict(transpose_b=tb, compute_kernel_config=ckc(fid))
            if pc is not None:
                kw["program_config"] = pc
            try:
                o = ttnn.matmul(a, b, **kw)
                err = float(((ttnn.to_torch(o)[:, :256].double() - ref).norm() / ref.norm()))
                ttnn.deallocate(o)
                ms, n0 = [], len(samples)
                for _ in range(A.reps):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    o = ttnn.matmul(a, b, **kw)
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o)
                clk = samples[n0:]
                med = statistics.median(ms)
                row["arms"][arm] = {"ms": med, "tflops": fl / med / 1e9, "rel_f64": err,
                                    "aiclk_min": [min(c[i] for c in clk) for i in range(len(clk_paths))] if clk else None}
            except Exception as e:
                row["arms"][arm] = {"error": str(e)[:300]}
            print(name, arm, row["arms"][arm], flush=True)
        for t in (a, b16, b8):
            ttnn.deallocate(t)
        res["shapes"][name] = row
        pathlib.Path(A.out).write_text(json.dumps(res, indent=1) + "\n")
    stop.set()


if __name__ == "__main__":
    main()
