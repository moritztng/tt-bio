"""Pair-sized `ttnn.linear(core_grid=...)` calls OpenDDE still runs on ttnn's own plan, against the tuned paths.

cs5 (9LLG, 640 tokens, WH 1000 MHz) priced four of them far off their roof: the recycle projection
`linear_no_bias_z_cycle` [1,640,640,384] @ [384,384] at 98.3 ms a call (0.03 of roof), the template
`linear_no_bias_z` [1,640,640,64] @ [64,384] at 16.9 ms, the MSA embed [1,1280,640,34] @ [34,128] at
9.0 ms, and the trimul row-blocked tail [1,128,1216,384] @ [384,384] at 9.8 ms.

Arms: ship (`ttnn.linear(core_grid=CORE_GRID_MAIN)`, what those sites run), flat (the same call on
[1,1,M,K]), pp (`_pair_proj_linear`, the trunk's tuned pair projection), mm (minimal_matmul).
nogrid (ttnn's plan without core_grid) and cfgN (`_pair_proj_config` at in0_block_w <= N:
bw 1 keeps the shipped K order, so it is the bit-exact candidate). Each arm reports torch.equal against ship and float64 rel_rms on every 37th row.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/plin_sweep.py OUT.json [--shapes 1x640x640x384:384:HiFi3 ...]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--shapes", nargs="+", default=["1x640x640x384:384:HiFi3", "1x640x640x64:384:HiFi3",
                                                 "1x1280x640x34:128:HiFi3", "1x128x1216x384:384:HiFi4"])
ap.add_argument("--ncall", type=int, default=4); ap.add_argument("--reps", type=int, default=3)
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
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = {"arch": str(dev.arch()), "shapes": {}}
for spec in a.shapes:
    dims, n, fid, *acc = spec.split(":")   # a 4th field "noacc" runs with fp32_dest_acc_en off (fast sets' acc_off)
    shape = [int(d) for d in dims.split("x")]
    K, N = shape[-1], int(n)
    M = 1
    for d in shape[:-1]:
        M *= d
    torch.manual_seed(0)
    xt = torch.randn(*shape).bfloat16()
    wt = (torch.randn(K, N) * K ** -0.5).bfloat16()
    x, w = up(xt), up(wt)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                           math_approx_mode=False, fp32_dest_acc_en=not acc, packer_l1_acc=True)
    rows = torch.arange(0, M, 37)
    ref = (xt.reshape(M, K)[rows].double() @ wt.double())

    def ship():
        return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                           compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)

    def flat():
        xf = ttnn.reshape(x, (1, 1, M, K))
        o = ttnn.linear(xf, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        return ttnn.reshape(o, shape[:-1] + [N])

    def pp():
        return T._pair_proj_linear(x, w, ckc, ttnn.bfloat16)

    def mm():
        return ttnn.experimental.minimal_matmul(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                dtype=ttnn.bfloat16, compute_kernel_config=ckc)

    def nogrid():
        return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                           compute_kernel_config=ckc)

    def cfg(cap):
        pc = T._pair_proj_config(x, w, bw_cap=cap)
        if pc is None:
            raise RuntimeError(f"_pair_proj_config declines at bw_cap {cap}")
        return lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                   compute_kernel_config=ckc, program_config=pc)

    arms = {"ship": ship, "flat": flat, "pp": pp, "mm": mm, "nogrid": nogrid}
    for cap in (1, 2, 4):
        try:
            arms[f"cfg{cap}"] = cfg(cap)
        except Exception as e:
            print(spec, f"cfg{cap}", str(e).splitlines()[0][:200], flush=True)
    r = res["shapes"][spec] = {}
    base = None
    ok = []
    for name, fn in arms.items():
        try:
            o = fn()
            ot = ttnn.to_torch(o).reshape(M, N)
            o.deallocate()
        except Exception as e:  # an L1 refusal or an unsupported shape
            r[name] = {"error": str(e).splitlines()[0][:200]}
            print(spec, name, r[name], flush=True)
            continue
        if base is None:
            base = ot
        d = ot[rows].double() - ref
        r[name] = {"equal": bool(torch.equal(ot, base)),
                   "rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())}
        ok.append(name)
        print(spec, name, r[name], flush=True)
    times = {nm: [] for nm in ok}
    for _ in range(a.reps):
        for nm in ok:
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(a.ncall):
                arms[nm]().deallocate()
            ttnn.synchronize_device(dev)
            times[nm].append((time.perf_counter() - t0) / a.ncall * 1e3)
    for nm, t in times.items():
        r[nm].update(ms_median=statistics.median(t), ms=t)
        print(f"{spec:28s} {nm:5s} {statistics.median(t):8.3f} ms", flush=True)
    x.deallocate(); w.deallocate()
json.dump(res, open(a.out, "w"), indent=1)
