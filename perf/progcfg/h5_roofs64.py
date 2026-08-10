#!/usr/bin/env python3
"""Roofs, core utilisation and an own-pass re-take of the isolated 0.172 ms/region, at the shape
the Q-A narrow wall is denominated against: [1,512,512,64] with a [64,64] weight.

The k-chunk pass measured the copy roofs at [1,512,512,256] and [1,298,320,256]. The template
track's own shape was never taken, so the narrow wall had no roof of its own. Charter 4.1: measure
it, never inherit it.
"""
import json, statistics as st, sys, time
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/protenix-trunk--z-h5-infold")
sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
BF = ttnn.bfloat16
R = {"host": "qb2", "chip": 0, "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
     "grid": list(T.COMPUTE_GRID_MAIN), "l1_bank_bytes": T._l1_bank_bytes(),
     "note": "qb2 / ttnn 0.68.0 -- every absolute is a RATIO owing a qb1/0.67.4 re-take"}


def med(fn, reps=7):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return st.median(ts) * 1e3


def mk(shape, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32), dtype=BF,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


# ---- copy roofs at the op's own shape, both destinations -----------------------------------------
roofs = {}
for tag, shape in (("512x512x64", (1, 512, 512, 64)), ("512x512x256", (1, 512, 512, 256))):
    mb = shape[1] * shape[2] * shape[3] * 2 / 1e6
    src = mk(shape)
    d = med(lambda: ttnn.clone(src, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=BF))
    l = med(lambda: ttnn.clone(src, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=BF))
    roofs[tag] = {"MB": round(mb, 2), "dram_ms": round(d, 4), "l1_ms": round(l, 4),
                  "dram_GBs": round(2 * mb / d, 1), "l1_GBs": round(2 * mb / l, 1),
                  "l1_over_dram": round(d / l, 3)}
    ttnn.deallocate(src)
    print(tag, roofs[tag], flush=True)
R["copy_roofs"] = roofs

# ---- core utilisation: read it off the program config the production helper actually picks -------
gx, gy = T.COMPUTE_GRID_MAIN
cores = {}
for tag, xs, ws in (("c64_l1", (1, 512, 512, 64), (64, 64)),
                    ("c256_l1", (1, 512, 512, 256), (256, 256))):
    x = mk(xs); w = mk(ws)
    for l1 in (True, False):
        cfg = T._pair_proj_config(x, w, bw_cap=T._PAIR_PROJ_BW, out_l1=l1) if l1 else \
              T._pair_proj_config(x, w)
        if cfg is None:
            cores[f"{tag}|out_l1={l1}"] = None
            continue
        mt = -(-xs[1] * xs[2] // 32)
        nt = -(-ws[-1] // 32)
        pm, pn = int(cfg.per_core_M), int(cfg.per_core_N)
        used = (-(-mt // pm)) * (-(-nt // pn))
        cores[f"{tag}|out_l1={l1}"] = {
            "m_tiles": mt, "n_tiles": nt, "per_core_M": pm, "per_core_N": pn,
            "in0_block_w": int(cfg.in0_block_w),
            "out_subblock_h": int(cfg.out_subblock_h), "out_subblock_w": int(cfg.out_subblock_w),
            "cores_engaged": min(used, gx * gy), "cores_available": gx * gy}
    ttnn.deallocate(x); ttnn.deallocate(w)
R["core_util"] = cores
print(json.dumps(cores, indent=1), flush=True)

# ---- own-pass re-take of the isolated region at [1,512,512,64] -----------------------------------
cells = {}
for arm in ("on", "off", "on", "off"):
    T._PAIR_PROJ_L1_OUT = (arm == "on")
    T._pair_proj_program_config.cache_clear()
    T._L1_OUT_REFUSED.clear()
    x = mk((1, 512, 512, 64)); xn = mk((1, 512, 512, 64))
    wp = mk((64, 64)); wg = mk((64, 64))
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           fp32_dest_acc_en=False, packer_l1_acc=True) \
        if not hasattr(ttnn, "BlackholeComputeKernelConfig") else \
        ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                               fp32_dest_acc_en=False, packer_l1_acc=True)

    def region():
        p = T._trimul_out_proj(x, wp, ckc)
        g = T._trimul_out_proj(xn, wg, ckc)
        out = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return out
    ms = med(region, reps=7)
    buf = None
    p = T._trimul_out_proj(x, wp, ckc)
    buf = "L1" if p.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
    ttnn.deallocate(p)
    cells.setdefault(arm, []).append(round(ms, 4))
    print("isolated region", arm, round(ms, 4), "branch", buf, flush=True)
    for t in (x, xn, wp, wg):
        ttnn.deallocate(t)
R["isolated_region_512x64_ms"] = cells
R["isolated_off_minus_on_ms"] = round(st.median(cells["off"]) - st.median(cells["on"]), 4)
R["isolated_aa_floor_on_ms"] = round(abs(cells["on"][0] - cells["on"][1]), 4)
print("OFF-ON", R["isolated_off_minus_on_ms"], "A/A floor", R["isolated_aa_floor_on_ms"], flush=True)

out = ROOT / "perf" / "progcfg" / "h5_roofs64_qb2c0.json"
out.write_text(json.dumps(R, indent=1))
print("wrote", out, flush=True)
