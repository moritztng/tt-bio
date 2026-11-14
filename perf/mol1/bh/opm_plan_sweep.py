#!/usr/bin/env python3
"""mol1-bh-prof: OPM contraction z = a b^T on Blackhole, ttnn's own plan against `opm_contract_config`, per K tile count.

ttnn's auto plan on a p150a (grid 11 wide) swings with Kt mod 11 (perf/spd_msa/zmm_kpad3.py), so the plan is priced across
depths, not at one. Shapes (--shapes): Mol-1's 1024-token row block (M 4096, N 32768) and the 256/512/1024-token whole calls.
bf16 HiFi4, fp32 dest acc, packer L1 acc (the fold's config). Median of --reps, AICLK of every card sampled during the calls,
rel L2 against float64 on the first 256 rows.

    TT_VISIBLE_DEVICES=<card> timeout 1800 python opm_plan_sweep.py out.json [--kt 64,192,...]
"""
import argparse, glob, json, os, pathlib, statistics, sys, threading, time
import torch

ROOT = pathlib.Path(os.environ.get("TT_BIO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
SHAPES = {"1024 block": (4096, 32768), "512 whole": (16384, 16384), "256 whole": (8192, 8192),
          "1024 whole": (32768, 32768)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--kt", default="32,64,96,128,186,187,188,192,197,198,199,256,310,311,318,319,320")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--shapes", default="1024 block,512 whole")
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    grid = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    clk_paths = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append([int(open(f).read()) for f in clk_paths])
            time.sleep(0.1)
    threading.Thread(target=poll, daemon=True).start()
    g = torch.Generator().manual_seed(0)
    res = {"host": os.uname().nodename, "sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "grid": [grid.x, grid.y], "clk_paths": clk_paths, "rows": []}
    for name in A.shapes.split(","):
        M, Nn = SHAPES[name]
        for kt in map(int, A.kt.split(",")):
            K = kt * 32
            a_t = torch.randn(M, K, generator=g).to(torch.bfloat16)
            b_t = (torch.randn(Nn, K, generator=g) * K ** -0.5).to(torch.bfloat16)
            ref = a_t[:256].double() @ b_t.double().T
            a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            row = {"shape": name, "M": M, "N": Nn, "kt": kt, "arms": {}}
            pc = TT.opm_contract_config(M // 32, Nn // 32, kt, grid)
            for arm, cfg in (("auto", None), ("plan", pc)):
                if arm == "plan" and cfg is None:
                    row["arms"][arm] = None
                    continue
                kw = dict(transpose_b=True, compute_kernel_config=ckc, program_config=cfg)
                try:
                    o = ttnn.matmul(a, b, **kw)
                    err = float((ttnn.to_torch(o)[:256].double() - ref).norm() / ref.norm())
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
                    row["arms"][arm] = {"ms": med, "tflops": 2.0 * M * K * Nn / med / 1e9, "rel_f64": err,
                                        "aiclk_max": [max(c[i] for c in clk) for i in range(len(clk_paths))] if clk else None}
                except Exception as e:
                    row["arms"][arm] = {"error": str(e)[:300]}
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            pathlib.Path(A.out).write_text(json.dumps(res, indent=1) + "\n")
    stop.set()


if __name__ == "__main__":
    main()
