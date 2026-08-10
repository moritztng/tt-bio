#!/usr/bin/env python3
"""No-fold probe for the _NARROW_PROJ_BW cap sweep: roofs, the config ladder, the speed ladder,
and the arithmetic parity of every cap against an fp32 reference.

THIS IS A THROWAWAY EXPERIMENT HARNESS. It changes nothing in tt_bio/; every cap is passed as the
`bw_cap` ARGUMENT that `_pair_proj_config` already takes, so the module default stays at 1.

WHAT IT ANSWERS THAT A FOLD CANNOT. The fold arms give ms/fold; they cannot give the arithmetic.
`_NARROW_PROJ_BW > 1` is not `torch.equal` against the production contraction order, so the org needs
to know which DIRECTION it moves, and against what. Here the reference is a torch fp32 matmul of the
same bf16 operands, so "cap 8 is not bit-exact" and "cap 8 is further from the truth" become separate
measurable claims. `k_tiles / in0_block_w` partial sums fold through `packer_l1_acc` per output tile
-- 8 of them at bw=1, exactly 1 at bw=8 -- so the a-priori expectation is that accuracy IMPROVES with
the cap while speed does too, and that there is no trade at all. This is where that is settled.

Usage (qb2 chip 0, ttnn 0.68.0, so every absolute is a RATIO -- charter 4.8):

  SP=~/tt-bio-dev/env/lib/python3.10/site-packages
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-narrowbw-512 \\
  TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \\
  PYTHONPATH=$PWD ~/tt-bio-dev/env/bin/python3 perf/narrowbw/nbw_probe.py \\
      --out perf/narrowbw/nbw_probe_qb2c0.json

NEVER run it without TT_VISIBLE_DEVICES and TT_MESH_GRAPH_DESC_PATH: the first ttnn tensor opens the
CLUSTER, not a chip, and on qb2 that means all four chips and a block on `CHIP_IN_USE_<n>` behind
whoever holds the board partner. Cost me two killed processes in the planning pass.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAPS = (1, 2, 4, 8, 16)

# The three narrow pair-track projection sites, at both sizes, with the call counts the sibling
# leg COUNTED in a live fold (state/protenix-trunk--z-survival-512.md 1.5 / 6.2): 484 + 240 + 40.
SITES = {
    "pairbias": {"w": (256, 16), "calls": 484},
    "pwa":      {"w": (256, 1),  "calls": 240},
    "template": {"w": (256, 64), "calls": 40},
}
XSHAPE = {512: (1, 512, 512, 256), 298: (1, 298, 320, 256)}


class Shim:
    """Shape/dtype stand-in so the config ladder needs no allocation."""

    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.padded_shape = tuple(shape)
        self.dtype = dtype


def cfg_fields(c):
    if c is None:
        return None
    return {"in0_block_w": c.in0_block_w, "out_subblock_h": c.out_subblock_h,
            "out_subblock_w": c.out_subblock_w, "out_block_h": c.out_block_h,
            "out_block_w": c.out_block_w, "per_core_M": c.per_core_M, "per_core_N": c.per_core_N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import ttnn
    import torch
    import tt_bio.tenstorrent as T
    import importlib.metadata as im

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    cores = gx * gy
    # Production's own kernel config (TorchWrapper): HiFi4, fp32 dest accumulate, packer_l1_acc.
    # A LoFi peak would be a roof no trunk op can reach, which charter 4.6 exists to prevent.
    ckc = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"), "grid": [gx, gy], "cores": cores,
           "l1_bank_bytes": T._l1_bank_bytes(),
           "per_core_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
           "narrow_proj_bw_default": str(T._NARROW_PROJ_BW),
           "pair_proj_bw": str(T._PAIR_PROJ_BW), "pair_proj_l1_bw": str(T._PAIR_PROJ_L1_BW),
           "note": "qb2 at ttnn 0.68.0 -- every absolute is a RATIO owing a qb1/0.67.4 re-take; "
                   "a ratio BETWEEN CAPS on one card is what this card is good for"}

    def timed(fn, reps):
        # One untimed warmup, because the FIRST touch of a new program config pays its JIT compile
        # inside the timed region: measured 0.4 s on a 0.7 ms op, which leaves the median intact and
        # makes max-min meaningless as a noise floor. The tell in the un-warmed run was cap 16's
        # spread of 0.10 ms against cap 8's 360 ms -- cap 16 is config-identical to cap 8, so its
        # kernel was already compiled and only it reported an honest spread.
        ttnn.synchronize_device(dev)
        w = fn()
        ttnn.synchronize_device(dev)
        if w is not None:
            ttnn.deallocate(w)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            if out is not None:
                ttnn.deallocate(out)
        return st.median(ts), (max(ts) - min(ts))

    # ---- A. the config ladder, no allocation: what each cap actually asks the kernel for ---------
    ladder = {}
    for size, xs in XSHAPE.items():
        for name, s in SITES.items():
            x, w = Shim(list(xs), ttnn.bfloat16), Shim(list(s["w"]), ttnn.bfloat16)
            row = {}
            for cap in CAPS:
                T._pair_proj_program_config.cache_clear()
                row[str(cap)] = cfg_fields(T._pair_proj_config(x, w, bw_cap=cap))
            ladder[f"{name}|{size}"] = row
    res["config_ladder"] = ladder

    # ---- B. roofs on this card, this pass (charter 4.1: never inherit one) -----------------------
    roofs = {}
    for size, xs in XSHAPE.items():
        nbytes = 1
        for d in xs:
            nbytes *= d
        nbytes *= 2
        src = ttnn.from_torch(torch.zeros(*xs, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row = {"MB": round(nbytes / 2 ** 20, 2)}
        for dst, mc in (("DRAM", ttnn.DRAM_MEMORY_CONFIG), ("L1", ttnn.L1_MEMORY_CONFIG)):
            try:
                ms, spread = timed(lambda mc=mc: ttnn.clone(src, memory_config=mc), a.reps)
                ms *= 1e3
                # A clone reads nbytes and writes nbytes: the copy roof is bidirectional bytes.
                row[f"clone_to_{dst}_ms"] = round(ms, 4)
                row[f"clone_to_{dst}_GBps"] = round(2 * nbytes / (ms * 1e-3) / 1e9, 1)
                row[f"clone_to_{dst}_spread_ms"] = round(spread * 1e3, 4)
            except Exception as e:                                                 # noqa: BLE001
                row[f"clone_to_{dst}_error"] = f"{type(e).__name__}: {e}"[:200]
        ttnn.deallocate(src)
        roofs[f"clone|{size}"] = row

    n = 4096
    ha = torch.randn(n, n, dtype=torch.bfloat16)
    ta = ttnn.from_torch(ha, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    tb = ttnn.from_torch(ha.t().contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    ms, spread = timed(lambda: ttnn.matmul(ta, tb, compute_kernel_config=ckc,
                                           core_grid=T.CORE_GRID_MAIN), a.reps)
    tflops = 2 * n ** 3 / (ms) / 1e12
    roofs["matmul_4096_prod_fidelity"] = {"ms": round(ms * 1e3, 4), "TFLOPs": round(tflops, 2),
                                          "spread_ms": round(spread * 1e3, 4),
                                          "fidelity": "HiFi4, fp32_dest_acc_en, packer_l1_acc"}
    ttnn.deallocate(ta)
    ttnn.deallocate(tb)
    dram = roofs["clone|512"].get("clone_to_DRAM_GBps")
    if dram:
        roofs["machine_balance_FLOP_per_byte"] = round(tflops * 1e12 / (dram * 1e9), 1)
    res["roofs"] = roofs

    # ---- C. the speed ladder and the arithmetic, per site, per size ------------------------------
    # One x per size, reused by all three sites and all five caps, so the read is the same bytes
    # every time and the only thing that moves is in0_block_w.
    sweeps, parity = {}, {}
    for size, xs in XSHAPE.items():
        torch.manual_seed(0)
        hx = torch.randn(*xs, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(hx, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xf = hx.reshape(-1, xs[-1]).float()          # the bf16 operand, exactly, upcast
        for name, s in SITES.items():
            torch.manual_seed(1)
            hw = torch.randn(*s["w"], dtype=torch.float32).to(torch.bfloat16)
            w = ttnn.from_torch(hw, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref = (xf @ hw.float())                  # fp32 reference over the SAME bf16 operands
            row, par, outs = {}, {}, {}

            def run(cfg):
                if cfg is None:
                    return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                                       core_grid=T.CORE_GRID_MAIN)
                return ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                                   program_config=cfg)

            for cap in list(CAPS) + [None]:
                T._pair_proj_program_config.cache_clear()
                cfg = None if cap is None else T._pair_proj_config(x, w, bw_cap=cap)
                key = "core_grid" if cap is None else str(cap)
                if cap is not None and cfg is None:
                    row[key] = {"error": "config REFUSED -- the helper would return None here, so "
                                         "the production call falls back to core_grid="}
                    continue
                try:
                    ms, spread = timed(lambda cfg=cfg: run(cfg), a.reps)
                except Exception as e:                                            # noqa: BLE001
                    row[key] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    continue
                out = run(cfg)
                ttnn.synchronize_device(dev)
                got = ttnn.to_torch(out).reshape(-1, s["w"][-1]).float()
                ttnn.deallocate(out)
                outs[key] = got
                d = (got - ref).abs()
                gc, rc = got.reshape(-1), ref.reshape(-1)
                pcc = float(torch.corrcoef(torch.stack([gc.double(), rc.double()]))[0, 1])
                m_tiles = xs[1] * -(-xs[2] // 32)
                row[key] = {"ms": round(ms * 1e3, 5), "spread_ms": round(spread * 1e3, 5),
                            "in0_block_w": (None if cfg is None else cfg.in0_block_w),
                            "cores_engaged": (None if cfg is None
                                              else -(-m_tiles // cfg.per_core_M)),
                            "of_cores": cores}
                par[key] = {"max_abs_vs_fp32": float(d.max()),
                            "rms_vs_fp32": float((d ** 2).mean().sqrt()),
                            "pcc_vs_fp32": pcc,
                            "packer_l1_acc_folds_per_out_tile":
                                (None if cfg is None else 8 // cfg.in0_block_w)}
            # torch.equal between every pair of caps, and against the core_grid reference: the
            # bit-exactness claim in the constant's comment is exactly this matrix.
            eq = {}
            ks = list(outs)
            for i, k1 in enumerate(ks):
                for k2 in ks[i + 1:]:
                    eq[f"{k1}=={k2}"] = bool(torch.equal(outs[k1], outs[k2]))
            sweeps[f"{name}|{size}"] = row
            parity[f"{name}|{size}"] = {"vs_fp32": par, "torch_equal": eq,
                                        "calls_per_fold": s["calls"]}
            ttnn.deallocate(w)
        ttnn.deallocate(x)
    res["speed_ladder"] = sweeps
    res["parity"] = parity

    # ---- D. ms/fold at the counted call numbers, so the table is in the scoreboard's currency ----
    perfold = {}
    for size in XSHAPE:
        tot = {}
        for cap in [str(c) for c in CAPS] + ["core_grid"]:
            s = 0.0
            ok = True
            for name, meta in SITES.items():
                r = sweeps[f"{name}|{size}"].get(cap, {})
                if "ms" not in r:
                    ok = False
                    break
                s += r["ms"] * meta["calls"]
            tot[cap] = round(s, 2) if ok else None
        base = tot.get("1")
        tot["saving_vs_cap1_ms_per_fold"] = {
            k: (round(base - v, 2) if (base is not None and v is not None) else None)
            for k, v in tot.items() if k != "1" and not isinstance(v, dict)}
        perfold[str(size)] = tot
    res["isolated_ms_per_fold"] = perfold
    res["ms_per_fold_note"] = ("764 counted calls = 484 pairbias + 240 pwa + 40 template "
                               "(z-survival-512's in-fold census, not an assumption). These are "
                               "ISOLATED-probe ms/fold and are an upper bound on the in-fold "
                               "figure; nbw_arms.py is the number of record.")

    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
