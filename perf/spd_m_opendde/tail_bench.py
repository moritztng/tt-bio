"""The trimul row-block tail projection (`_tail_rows`) at OpenDDE's refiner shape, `core_grid=` vs the `tail_wide`
plan (`_narrow_proj_linear`'s wide leg), per weight dtype and trunk fidelity: ms/call, torch.equal, rel_rms vs float64.

cs10 (9LLG 640, both WH modes): [1,128,1216,384] @ [384,384] at 9.85 ms x 144 calls per fold, 0.10 of roof.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/tail_bench.py OUT.json [--rows 128] [--seq 1216]
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rows", type=int, default=128); ap.add_argument("--seq", type=int, default=1216)
ap.add_argument("--c", type=int, default=384); ap.add_argument("--ncall", type=int, default=10); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
torch.manual_seed(0)
hx = torch.randn(1, a.rows, a.seq, a.c)
hw = torch.randn(a.c, a.c) / a.c ** 0.5
x = ttnn.from_torch(hx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
sel = slice(0, a.rows, 7)
x64 = ttnn.to_torch(x)[:, sel].double()
kt = -(-a.c // 32)
cfg_any = T._pair_proj_program_config(a.rows * -(-a.seq // 32), kt, kt, 1, 2, False, None, 0, False)
res = {"shape": [1, a.rows, a.seq, a.c], "arch": str(dev.arch()), "arms": {}}
for wdt in ("bf16", "bfp8"):
    w = ttnn.from_torch(hw, dtype=ttnn.bfloat16 if wdt == "bf16" else ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = x64 @ ttnn.to_torch(w).double()
    for fid in ("HiFi4", "HiFi2"):   # the refiner runs on the base HiFi4 config (refiner_trunk_ckc killed)
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        arms = {
            "core_grid": lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                             compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN),
            "wide": lambda: T._narrow_proj_linear(x, w, ckc, ttnn.bfloat16),
            # the same plan without `_pair_proj_config`'s bf16-weights gate (fast mode's bfp8 weights)
            "wide_any": lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                            compute_kernel_config=ckc, program_config=cfg_any),
        }
        outs = {}
        for name, f in arms.items():
            key = f"{wdt}/{fid}/{name}"
            o = f()
            if o is None:
                res["arms"][key] = {"declined": True}
                continue
            full = ttnn.to_torch(o); o.deallocate()
            outs[name] = full
            got = full[:, sel].double()
            ms = []
            for _ in range(a.reps):
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(a.ncall):
                    f().deallocate()
                ttnn.synchronize_device(dev)
                ms.append((time.perf_counter() - t0) / a.ncall * 1e3)
            res["arms"][key] = {"ms": ms, "ms_min": min(ms),
                                "rel_rms": float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())}
        for name in ("wide", "wide_any"):
            if name in outs and "core_grid" in outs:
                res["arms"][f"{wdt}/{fid}/{name}"]["equal_core_grid"] = bool(torch.equal(outs[name], outs["core_grid"]))
        print(json.dumps({k: v for k, v in res["arms"].items() if k.startswith(f"{wdt}/{fid}")}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
