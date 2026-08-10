#!/usr/bin/env python3
"""No-fold probe: the size cliff of every gated flag, this card's roofs, and the two live mechanisms.

Throwaway experiment harness, not production code. Nothing in `tt_bio/` changes.

Three products, one device open, no fold:

A. THE CLIFF, PER FLAG, FROM THE PRODUCTION HELPERS THEMSELVES. `_l1_memory_config_if_it_fits` is a
   pure function of (padded shape, dtype, live per-core L1, grid), so evaluating it on the padded
   shape at every N gives the exact token count where each flag stops firing -- on THIS card, with
   THIS card's budget read live, not a shape argument. Charter 4.10 requires every win to carry the
   size at which it stops paying; this is that table. The in-fold census in `surv_arms.py` is still
   the ground truth at 512 aa: this extends it to the neighbouring sizes for free.

B. THE ROOFS, MEASURED HERE THIS PASS (charter 4.1, never inherited). `ttnn.clone` at both pair
   shapes, DRAM destination and L1 destination, plus a square matmul peak so the machine balance is
   this card's own number rather than the other card's 338 FLOP/byte.

C. THE TWO MECHANISMS THE 512 aa ARMS TURN ON, PRICED IN ISOLATION. The narrow pair-track projection
   [1,N,N,256] @ [256,8] in three forms: production (DRAM source, tuned config), the L1-source form
   the norm flags would hand it if the fit test passed, and the `_NARROW_PROJ_BW=None` core_grid form
   that is the pre-X2 baseline. If the L1-source form throws a circular-buffer clash, that throw is
   the mechanism behind the 1.5x headroom and is reported as a result, not as an error.

Usage:
  SP=~/tt-bio-dev/env/lib/python3.10/site-packages
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-survival-512 \\
  TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \\
  ~/tt-bio-dev/env/bin/python3 perf/survival512/surv_envelope.py --out perf/survival512/surv_envelope_qb2c0.json
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Shim:
    """Just enough tensor for the production fit helpers: they read shape and dtype only."""

    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.padded_shape = tuple(shape)
        self.dtype = dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import importlib.metadata as im
    import torch

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    # Production's own kernel config (tenstorrent.py TorchWrapper): HiFi4 with fp32 dest accumulate.
    # A LoFi peak would be a roof no trunk op can reach, which charter 4.6 exists to prevent.
    ckc = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"), "grid": [gx, gy],
           "cores": gx * gy, "per_core_l1_unreserved": per_core,
           "l1_fit_budget_bytes": per_core * gx * gy, "l1_bank_bytes": T._l1_bank_bytes(),
           "note": "qb2 at ttnn 0.68.0 -- every absolute is a RATIO; the cliff sizes and the "
                   "L1/DRAM ratio are the transferable quantities",
           "flags": {k: str(getattr(T, k)) for k in
                     ("_PAIR_PROJ_L1_OUT", "_PAIR_BIAS_L1_NORM", "_PWA_L1_NORM",
                      "_TEMPLATE_L1_NORM", "_NARROW_PROJ_BW", "_PAIR_PROJ_BW", "_PAIR_PROJ_L1_BW")}}

    def timed(fn, reps):
        ttnn.synchronize_device(dev)
        out = None
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            if out is not None:
                ttnn.deallocate(out)
        return st.median(ts)

    # ---- A. the cliff, per flag class -------------------------------------------------------------
    cliff = []
    for n in range(288, 801, 32):
        row = {"n_padded": n}
        for c in (256, 64):
            t = Shim([1, n, n, c], ttnn.bfloat16)
            bytes_ = n * n * c * 2
            row[f"c{c}_MB"] = round(bytes_ / 2 ** 20, 2)
            row[f"c{c}_norm_1.5x"] = (
                "L1" if T._l1_memory_config_if_it_fits(t, 1.5) is ttnn.L1_MEMORY_CONFIG else "DRAM")
            row[f"c{c}_transpose_2.5x"] = (
                "L1" if T._l1_memory_config_if_it_fits(t, 2.5) is ttnn.L1_MEMORY_CONFIG else "DRAM")
            w8 = Shim([c, 8], ttnn.bfloat16)
            wc = Shim([c, c], ttnn.bfloat16)
            row[f"c{c}_narrow_cfg"] = T._pair_proj_config(t, w8, bw_cap=T._NARROW_PROJ_BW) is not None
            row[f"c{c}_projl1_cfg"] = T._pair_proj_config(t, wc, bw_cap=T._PAIR_PROJ_L1_BW,
                                                          out_l1=True) is not None
        cliff.append(row)
    res["cliff"] = cliff

    def edge(key, want="L1"):
        """Largest padded n at which `key` still reads `want`, and the first n at which it does not."""
        last = None
        for r in cliff:
            v = r[key]
            v = ("L1" if v else "DRAM") if isinstance(v, bool) else v
            if v == want:
                last = r["n_padded"]
            elif last is not None:
                return {"last_" + want: last, "first_other": r["n_padded"]}
        return {"last_" + want: last, "first_other": None}

    res["cliff_edges"] = {k: edge(k) for k in
                          ("c256_norm_1.5x", "c256_transpose_2.5x", "c256_narrow_cfg",
                           "c256_projl1_cfg", "c64_norm_1.5x", "c64_transpose_2.5x",
                           "c64_narrow_cfg", "c64_projl1_cfg")}

    # ---- B. roofs on this card, this pass ---------------------------------------------------------
    roofs = {}
    for n, c in ((512, 256), (512, 64)):
        shape = (1, n, n, c)
        nbytes = n * n * c * 2
        host = torch.zeros(shape, dtype=torch.bfloat16)
        src = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row = {"MB": round(nbytes / 2 ** 20, 2)}
        for dst, mc in (("DRAM", ttnn.DRAM_MEMORY_CONFIG), ("L1", ttnn.L1_MEMORY_CONFIG)):
            try:
                ms = timed(lambda mc=mc: ttnn.clone(src, memory_config=mc), a.reps) * 1e3
                row[f"clone_to_{dst}_ms"] = round(ms, 4)
                row[f"clone_to_{dst}_GBps"] = round(2 * nbytes / (ms * 1e-3) / 1e9, 1)
            except Exception as e:                                             # noqa: BLE001
                row[f"clone_to_{dst}_ms"] = f"{type(e).__name__}: {str(e)[:160]}"
        if isinstance(row.get("clone_to_DRAM_GBps"), float) and isinstance(row.get("clone_to_L1_GBps"), float):
            row["L1_over_DRAM"] = round(row["clone_to_L1_GBps"] / row["clone_to_DRAM_GBps"], 3)
        ttnn.deallocate(src)
        roofs[f"[1,{n},{n},{c}]"] = row
    # square matmul peak -> this card's machine balance
    k = 4096
    ha = torch.randn(k, k, dtype=torch.bfloat16)
    A = ttnn.from_torch(ha, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    B_ = ttnn.from_torch(ha, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ms = timed(lambda: ttnn.matmul(A, B_, compute_kernel_config=ckc,
                                   core_grid=ttnn.CoreGrid(x=gx, y=gy)), a.reps) * 1e3
    peak = 2 * k ** 3 / (ms * 1e-3) / 1e12
    roofs["matmul_peak"] = {"shape": f"{k}^3 bf16", "ms": round(ms, 4), "TFLOPs": round(peak, 2),
                            "fidelity": "HiFi4, fp32_dest_acc_en, packer_l1_acc -- production's own "
                                        "kernel config, so this is the reachable peak and not a "
                                        "LoFi number no trunk op can see"}
    dram = roofs["[1,512,512,256]"].get("clone_to_DRAM_GBps")
    if isinstance(dram, float):
        roofs["machine_balance_FLOP_per_byte"] = round(peak * 1e12 / (dram * 1e9), 1)
    ttnn.deallocate(A)
    ttnn.deallocate(B_)
    res["roofs"] = roofs

    # ---- C. the two live mechanisms, priced in isolation ------------------------------------------
    mech = {}
    for n in (320, 512):
        shape = (1, n, n, 256)
        nbytes = n * n * 256 * 2
        x_dram = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                 device=dev, dtype=ttnn.bfloat16,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(256, 8, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        ck = ckc
        row = {"MB_source": round(nbytes / 2 ** 20, 2),
               "MB_out_padded": round(n * n * 32 * 2 / 2 ** 20, 2)}
        try:
            row["prod_dram_source_ms"] = round(timed(
                lambda: T._narrow_proj_linear(x_dram, w, ck, ttnn.bfloat16, l1_out=False),
                a.reps) * 1e3, 4)
        except Exception as e:                                                 # noqa: BLE001
            row["prod_dram_source_ms"] = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            x_l1 = ttnn.clone(x_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
            row["l1_source_ms"] = round(timed(
                lambda: T._narrow_proj_linear(x_l1, w, ck, ttnn.bfloat16, l1_out=True),
                a.reps) * 1e3, 4)
            ttnn.deallocate(x_l1)
        except Exception as e:                                                 # noqa: BLE001
            row["l1_source_ms"] = f"{type(e).__name__}: {str(e)[:200]}"
        try:
            row["core_grid_baseline_ms"] = round(timed(
                lambda: ttnn.linear(x_dram, w, compute_kernel_config=ck,
                                    core_grid=T.CORE_GRID_MAIN), a.reps) * 1e3, 4)
        except Exception as e:                                                 # noqa: BLE001
            row["core_grid_baseline_ms"] = f"{type(e).__name__}: {str(e)[:160]}"
        cfg = T._pair_proj_config(x_dram, w, bw_cap=T._NARROW_PROJ_BW)
        row["prod_program_config"] = str(cfg)[:400] if cfg is not None else None
        ttnn.deallocate(x_dram)
        ttnn.deallocate(w)
        mech[f"narrow_proj_[1,{n},{n},256]@[256,8]"] = row
    res["mechanisms"] = mech

    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
