#!/usr/bin/env python3
"""p3-l1-source-068 — X7's three arms re-taken at ttnn 0.68.0, plus the two sites it left.

Reuses `perf/p3l1/p3_l1_probe.py` verbatim for the `pair`, `trimul` and `site2` arms so the
0.67.4 and 0.68.0 columns are the same code, and adds:

  pwa       `tenstorrent.py:3074/3084` — one layer_norm(z) feeding EIGHT [256,1] head
            projections. The shared object here is the PRODUCER, not the consumer, so the
            per-call read saving is honestly 8x per invocation while the norm's own removed
            write is paid once. The arm measures the whole eight-head region, not a call.
  template  `protenix.py:2033` — one layer_norm(z3) feeding FOUR [256,64] projections inside
            `for t in range(nt)`. Same shape, nt=4 consumers, 40 calls/fold.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "p3l1"))
import tt_bio.tenstorrent as T          # noqa: E402
import p3_l1_probe as X7                # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TOK, NPAD, C_Z = X7.TOK, X7.NPAD, X7.C_Z
timed, cfg, l1_free = X7.timed, X7.cfg, X7.l1_free


def _norm_and_projections(dev, ckc, gx, gy, res, key, n_out_tiles, n_consumers, post):
    """One layer_norm(z) feeding `n_consumers` narrow projections of `n_out_tiles` tiles.

    `post` is the op production runs on each projection's result, so the arm prices the region
    the fold actually executes rather than a bare matmul.
    """
    z = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    nw = ttnn.from_torch(torch.randn(C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=DRAM)
    nb = ttnn.from_torch(torch.randn(C_Z) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=DRAM)
    ws = [ttnn.from_torch(torch.randn(C_Z, n_out_tiles * 32 if n_out_tiles > 1 else 1),
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=DRAM) for _ in range(n_consumers)]
    m_tiles, k_tiles = TOK * (NPAD // 32), C_Z // 32
    res[key + "_meta"] = {
        "in0_bytes": TOK * NPAD * C_Z * 2, "consumers": n_consumers,
        "out_bytes_each": TOK * NPAD * n_out_tiles * 32 * 2,
        "norm_calls_per_region": 1, "l1_free_before": l1_free(dev)}

    legs, outs = {}, {}
    #  label                norm mem, (bw, obh) or None, proj out mem
    plan = (("prod", DRAM, (1, 5), DRAM),
            ("prod_cg", DRAM, None, DRAM),
            ("outL1", DRAM, (1, 5), L1),
            ("normL1", L1, (1, 5), DRAM),
            ("normL1_outL1", L1, (1, 5), L1),
            ("normL1_cg", L1, None, DRAM))
    for lbl, nmem, bwobh, omem in plan:
        pc = None
        if bwobh is not None:
            pc = cfg(gx, gy, m_tiles, k_tiles, n_out_tiles, bwobh[0], bwobh[1], omem is L1)
            if pc is None:
                legs[lbl] = {"err": "config refused by the L1 budget"}
                print(f"  {lbl:16s} REFUSED by the L1 budget", flush=True)
                continue

        def one(zn, w):
            if pc is None:
                b = ttnn.linear(zn, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                memory_config=omem, dtype=ttnn.bfloat16)
            else:
                b = ttnn.linear(zn, w, compute_kernel_config=ckc, program_config=pc,
                                memory_config=omem, dtype=ttnn.bfloat16)
            return b

        def region():
            zn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=nmem)
            for w in ws:
                r = post(one(zn, w))
                ttnn.deallocate(r)
            ttnn.deallocate(zn)

        def parts():
            """norm, then ONE projection, then its post op, each synchronised."""
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            zn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=nmem)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            b = one(zn, ws[0])
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            r = post(b)
            ttnn.synchronize_device(dev)
            t3 = time.perf_counter()
            ttnn.deallocate(zn)
            return r, (t1 - t0), (t2 - t1), (t3 - t2)

        row = {}
        try:
            row["region_us"] = round(timed(region, dev, warm=2, pipe=2, reps=7) * 1e6, 2)
            nm, pj, po = [], [], []
            for _ in range(5):
                r, a, b_, c = parts()
                nm.append(a * 1e6); pj.append(b_ * 1e6); po.append(c * 1e6)
                ttnn.deallocate(r)
            row["norm_us"] = round(st.median(nm), 2)
            row["proj_us"] = round(st.median(pj), 2)
            row["post_us"] = round(st.median(po), 2)
            r, *_ = parts()
            outs[lbl] = ttnn.to_torch(r)
            ttnn.deallocate(r)
            zn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=nmem)
            live = [one(zn, w) for w in ws]
            ttnn.synchronize_device(dev)
            row["l1_free_all_consumers_live"] = l1_free(dev)
            for t in live:
                ttnn.deallocate(t)
            ttnn.deallocate(zn)
        except Exception as e:                                                # noqa: BLE001
            row["err"] = str(e)[:200]
        legs[lbl] = row
        print(f"  {lbl:16s} region {row.get('region_us','-'):>9} us  norm {row.get('norm_us','-'):>8}"
              f"  proj {row.get('proj_us','-'):>8}  post {row.get('post_us','-'):>8}"
              f"  {row.get('err','')}", flush=True)

    ref = outs.get("prod_cg")
    for lbl, row in legs.items():
        o = outs.get(lbl)
        if ref is not None and o is not None:
            row["torch_equal_vs_prod_cg"] = bool(torch.equal(ref, o))
            row["max_abs_vs_prod_cg"] = float((ref.double() - o.double()).abs().max())
    res[key] = legs
    for t in [z, nw, nb, *ws]:
        ttnn.deallocate(t)


def arm_pwa(dev, ckc, gx, gy, res):
    """tenstorrent.py:3074/3084 — 8 heads, [256,1] each, each followed by permute(2,0,1)."""
    _norm_and_projections(dev, ckc, gx, gy, res, "pwa", 1, 8,
                          lambda b: ttnn.permute(ttnn.reshape(b, tuple(b.shape)[1:]), (2, 0, 1)))


def arm_template(dev, ckc, gx, gy, res):
    """protenix.py:2033 — 4 templates, [256,64] each, each followed by the ttnn.add(tpl_a, .)."""
    add_to = {}

    def post(b):
        k = tuple(b.shape)
        if k not in add_to:
            add_to[k] = ttnn.from_torch(torch.randn(*k) * 0.1, dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=b.device(),
                                        memory_config=DRAM)
        return ttnn.add(add_to[k], b, memory_config=DRAM)

    _norm_and_projections(dev, ckc, gx, gy, res, "template", 2, 4, post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    choices=["pair", "trimul", "site2", "pwa", "template"])
    ap.add_argument("--card", default="qb1 card 2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN          # read AFTER the device is open
    dg = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    res = {"card": a.card, "ttnn": getattr(ttnn, "__version__", None) or __import__("importlib.metadata", fromlist=["version"]).version("ttnn"),
           "compute_grid_main_after_device_open": [gx, gy],
           "device_compute_with_storage_grid": [dg.x, dg.y],
           "core_grid_main": f"{T.CORE_GRID_MAIN.x}x{T.CORE_GRID_MAIN.y}",
           "l1_bank_bytes": T._l1_bank_bytes(),
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size())}
    print(json.dumps(res), flush=True)
    arms = {"pair": X7.arm_pair, "trimul": X7.arm_trimul, "site2": X7.arm_site2,
            "pwa": arm_pwa, "template": arm_template}
    for arm in a.arm:
        print(f"=== arm {arm} ===", flush=True)
        arms[arm](dev, ckc, gx, gy, res)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
