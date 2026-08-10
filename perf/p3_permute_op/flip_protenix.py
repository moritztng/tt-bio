#!/usr/bin/env python3
"""y-permute-flip deliverable 2: the flip's evidence on protenix-v2, re-taken on origin/main.

One process, one device open, model loaded once, `hoist=True` so the timed region is `model.fold`.
The control arm is unmodified production (`reblock_permute` disabled, which is what main runs today),
restored in-session after every round. Order:

  roofs on THIS card -> gate window + torch.equal -> cold fold per arm -> A/A control -> paired A/B
  -> trimul block wall -> interaction checks

Every stage is flushed to the output JSON as it completes, so a kill mid-run keeps what ran.
Load average is read beside every timed fold.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np
import torch
import ttnn

OUT = Path(__file__).resolve().parent
R: dict = {}
OUTPATH = None


def flush():
    if OUTPATH:
        Path(OUTPATH).write_text(json.dumps(R, indent=2, default=str) + "\n")


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


_L1_FIELDS = ("total_bytes_per_bank", "total_bytes_allocated_per_bank",
              "total_bytes_free_per_bank", "largest_contiguous_bytes_free_per_bank", "num_banks")


def l1_stats(dev):
    try:
        v = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    except Exception as e:                                            # noqa: BLE001
        return {"error": repr(e)[:160]}
    return {k: int(getattr(v, k)) for k in _L1_FIELDS if hasattr(v, k)} or {"repr": str(v)[:300]}


def mc(bt):
    return ttnn.DRAM_MEMORY_CONFIG if bt == "dram" else ttnn.L1_MEMORY_CONFIG


def mktensor(dev, N, C, bt, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, N, N, C, generator=g).to(torch.bfloat16)
    return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                              device=dev, memory_config=mc(bt))


def timed(fn, dev, reps, warm=2):
    """Throughput mode: enqueue `reps` calls, sync once. Sync on BOTH sides of the timed region."""
    outs = []
    for _ in range(warm):
        outs.append(fn())
    ttnn.synchronize_device(dev)
    for o in outs:
        ttnn.deallocate(o)
    outs = []
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        outs.append(fn())
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps * 1e6
    for o in outs:
        ttnn.deallocate(o)
    return dt


# --------------------------------------------------------------------------------------------------
def roofs(dev, RP):
    """Copy floors and the two permute walls at the fold's own tensor, measured on THIS card."""
    N, C = 298, 64
    bytes_one_way = N * N * C * 2
    out = {"shape": [1, N, N, C], "bytes_one_way": bytes_one_way, "load": load()}
    for src in ("l1", "dram"):
        for dst in ("l1", "dram"):
            _, x = mktensor(dev, N, C, src)
            reps = 4 if dst == "l1" else 8
            try:
                us = timed(lambda: ttnn.clone(x, memory_config=mc(dst)), dev, reps)
                out[f"clone_{src}_to_{dst}_us"] = round(us, 2)
                out[f"clone_{src}_to_{dst}_GBs"] = round(bytes_one_way / (us * 1e-6) / 1e9, 1)
            except Exception as e:                                    # noqa: BLE001
                out[f"clone_{src}_to_{dst}_us"] = f"REFUSED {repr(e)[:120]}"
            ttnn.deallocate(x)
    for bt in ("l1", "dram"):
        _, x = mktensor(dev, N, C, bt)
        reps = 4 if bt == "l1" else 8
        RP.set_enabled(True)
        out[f"kernel_{bt}_us"] = round(timed(lambda: RP.reblock_permute(x, mc(bt)), dev, reps), 2)
        RP.set_enabled(False)
        out[f"stock_permute_{bt}_us"] = round(
            timed(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc(bt)), dev, reps), 2)
        ttnn.deallocate(x)
    # core utilisation, read out of the work split the kernel actually uses
    Nt = (N + 31) // 32
    g = dev.compute_with_storage_grid_size()
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                             ttnn.CoreCoord(g.x - 1, g.y - 1))])
    (_, core_grid, cg1, cg2, w1, w2) = ttnn.split_work_to_cores(allc, Nt * Nt)
    n_cores = sum((cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
                  for cr in core_grid.ranges())
    out["num_groups"] = Nt * Nt
    out["cores_engaged"] = n_cores
    out["cores_total"] = g.x * g.y
    out["per_core_work"] = [w1, w2]
    k_l1, k_dr = out["kernel_l1_us"], out["kernel_dram_us"]
    out["kernel_GBs_l1"] = round(bytes_one_way / (k_l1 * 1e-6) / 1e9, 1)
    out["kernel_dram_minus_l1_us"] = round(k_dr - k_l1, 2)
    out["stock_dram_minus_l1_us"] = round(out["stock_permute_dram_us"] - out["stock_permute_l1_us"], 2)
    out["clone_dram_minus_l1_us"] = round(
        out["clone_dram_to_dram_us"] - out["clone_l1_to_l1_us"], 2)
    return out


def window(dev, RP):
    """Gate window ratios + torch.equal at every cell, on this grid at both chunk widths."""
    rows = []
    for N in (256, 288, 298, 320, 352, 384):
        for C in (32, 64):
            for bt in ("l1", "dram"):
                tref, x = mktensor(dev, N, C, bt)
                reps = 4 if bt == "l1" else 8
                try:
                    RP.set_enabled(True)
                    elig = RP.eligible(x, mc(bt))
                    kus = timed(lambda: RP.reblock_permute(x, mc(bt)), dev, reps)
                    yk = RP.reblock_permute(x, mc(bt))
                    RP.set_enabled(False)
                    sus = timed(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc(bt)),
                                dev, reps)
                    ys = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc(bt))
                    eq = bool(torch.equal(ttnn.to_torch(yk), ttnn.to_torch(ys)))
                    ttnn.deallocate(yk); ttnn.deallocate(ys)
                    rows.append({"N": N, "C": C, "buffer": bt, "eligible_by_gate": bool(elig),
                                 "kernel_us": round(kus, 2), "stock_us": round(sus, 2),
                                 "ratio_stock_over_kernel": round(sus / kus, 3),
                                 "torch_equal": eq})
                except Exception as e:                                # noqa: BLE001
                    RP.set_enabled(False)
                    rows.append({"N": N, "C": C, "buffer": bt, "error": repr(e)[:160]})
                ttnn.deallocate(x)
                print("win:", rows[-1], flush=True)
    return rows


# --------------------------------------------------------------------------------------------------
def main() -> int:
    global OUTPATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa-rounds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--block-reps", type=int, default=15)
    ap.add_argument("--out", default=str(OUT / "flip_protenix.json"))
    a = ap.parse_args()
    OUTPATH = a.out

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "w6_gate_msa"

    R.update({"wheel": "0.67.4", "host": "qb1 (tt-quietbox)",
              "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1")),
              "baseline": "origin/main 7224ff34",
              "reblock_permute_default_constant": RP.REBLOCK_PERMUTE,
              "enabled_at_import": RP._ENABLED,
              "aa": [], "rounds": []})
    flush()

    # interaction check: every L1 layer_norm that fell back to DRAM
    norm = {"l1": 0, "dram": 0}
    _orig_l1ln = T._l1_layer_norm

    def _counting_l1ln(x, headroom, **kw):
        t, in_l1 = _orig_l1ln(x, headroom, **kw)
        norm["l1" if in_l1 else "dram"] += 1
        return t, in_l1

    T._l1_layer_norm = _counting_l1ln

    trimul_throws = []
    _orig_tm = T.TriangleMultiplication.__call__
    grab: dict = {}
    l1_probe: dict = {}

    def _wrapped_tm(self, x, mask=None):
        if "mod" not in grab and int(x.shape[1]) == int(x.shape[2]) and int(x.shape[3]) >= 128:
            grab["mod"] = self
            grab["x"] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            grab["mask"] = (ttnn.clone(mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            if mask is not None else None)
            grab["shape"] = [int(d) for d in x.shape]
        if l1_probe.get("arm") is not None and "inside_trimul" not in l1_probe:
            l1_probe["inside_trimul"] = l1_stats(T.get_device())
        try:
            return _orig_tm(self, x, mask)
        except Exception as e:                                        # noqa: BLE001
            trimul_throws.append(repr(e)[:300])
            raise

    T.TriangleMultiplication.__call__ = _wrapped_tm

    from tt_baseline import build_fold
    t0 = time.perf_counter()
    one_fold, meta, state = build_fold("protenix-v2", msa_dir, target, a3m, hoist=True)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    coords: list = []
    _orig_fold = state.model.fold

    def _capture_fold(*ar, **kw):
        r = _orig_fold(*ar, **kw)
        c = r[0] if isinstance(r, tuple) else r
        coords.append(np.asarray(torch.as_tensor(c).detach().to(torch.float64).cpu()))
        return r

    state.model.fold = _capture_fold

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    R["meta"] = {"hardware": meta["hardware"], "card_type": meta.get("card_type"),
                 "aiclk_mhz": meta.get("aiclk_mhz"), "load_s": meta["load_s"],
                 "n_msa": meta["n_msa"]}
    R["grid"] = {"compute_grid_main": list(T.COMPUTE_GRID_MAIN),
                 "device_grid": [g.x, g.y],
                 "trimul_chunk_size_298_128": T._trimul_chunk_size(298, 128),
                 "triangle_mul_memory_config_298": str(T._triangle_mul_memory_config(298).buffer_type),
                 "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
                 "l1_idle": l1_stats(dev)}
    print("grid:", R["grid"], flush=True)
    flush()

    R["roofs"] = roofs(dev, RP)
    print("roofs:", R["roofs"], flush=True)
    flush()

    R["window"] = window(dev, RP)
    flush()

    def transpose_branch():
        t = ttnn.from_torch(torch.zeros(298, 320, 256, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = str(T._transpose_memory_config(t).buffer_type)
        ttnn.deallocate(t)
        return b

    R["transpose_memory_config_before"] = transpose_branch()

    def refused():
        return sorted(str(k)[:160] for k in T._L1_OUT_REFUSED)

    # --- cold fold per arm ------------------------------------------------------------------------
    RP.set_enabled(False)
    c0, m0 = one_fold()
    R["cold_base"] = {"s": round(c0, 3), "plddt": m0.get("plddt"),
                      "l1_out_refused": refused(), "load": load()}
    print("cold_base:", R["cold_base"], flush=True)
    l1_probe["arm"] = "wire"
    RP.set_enabled(True)
    n0 = RP.STATS[0]
    c1, m1 = one_fold()
    RP.set_enabled(False)
    l1_probe["arm"] = None
    R["cold_wire"] = {"s": round(c1, 3), "plddt": m1.get("plddt"),
                      "eligible_calls_served": RP.STATS[0] - n0,
                      "n_tokens": m1.get("n_tokens"), "l1_out_refused": refused(),
                      "trimul_throws": list(trimul_throws),
                      "norm_l1_vs_dram": dict(norm),
                      "l1_inside_trimul_wire_arm": l1_probe.get("inside_trimul"),
                      "load": load()}
    print("cold_wire:", R["cold_wire"], flush=True)
    R["transpose_memory_config_after_wire_fold"] = transpose_branch()
    flush()

    # --- A/A control -----------------------------------------------------------------------------
    for r in range(a.aa_rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        ta, _ = one_fold()
        la = load()
        RP.set_enabled(False)
        tb, _ = one_fold()
        row = {"round": r, "a_s": round(ta, 4), "b_s": round(tb, 4),
               "apparent_delta_ms": round((ta - tb) * 1e3, 1),
               "eligible_calls": RP.STATS[0] - n0, "load_a": la, "load_b": load()}
        R["aa"].append(row)
        print("aa:", row, flush=True)
        flush()
    aa = [x["a_s"] for x in R["aa"]] + [x["b_s"] for x in R["aa"]]
    # --aa-rounds 0 is the normal way to reach the block wall without paying for the fold arms, so
    # an empty aa list is expected, not an error. It used to raise ValueError out of max().
    R["aa_summary"] = {
        "max_abs_apparent_delta_ms": round(max((abs(x["apparent_delta_ms"]) for x in R["aa"]),
                                               default=0.0), 1),
        "all_folds_spread_ms": round((max(aa, default=0.0) - min(aa, default=0.0)) * 1e3, 1),
        "folds": [round(v, 4) for v in aa]}
    print("aa_summary:", R["aa_summary"], flush=True)
    flush()

    # --- the A/B ---------------------------------------------------------------------------------
    for r in range(a.rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        tb, mb = one_fold()
        n_base = RP.STATS[0] - n0
        lb = load()
        RP.set_enabled(True)
        n1 = RP.STATS[0]
        tw, mw = one_fold()
        n_wire = RP.STATS[0] - n1
        RP.set_enabled(False)
        row = {"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
               "delta_ms": round((tb - tw) * 1e3, 1),
               "eligible_calls_base": n_base, "eligible_calls_wire": n_wire,
               "plddt_base": mb.get("plddt"), "plddt_wire": mw.get("plddt"),
               "l1_out_refused_n": len(T._L1_OUT_REFUSED),
               "trimul_throws_n": len(trimul_throws),
               "load_base": lb, "load_wire": load()}
        R["rounds"].append(row)
        print("ab:", row, flush=True)
        flush()

    base = [x["base_s"] for x in R["rounds"]]
    wire = [x["wire_s"] for x in R["rounds"]]
    R["rejects"] = {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()}
    # --rounds 0 reaches the block wall without paying for the fold A/B arms, so an empty rounds
    # list is expected. Emitting a zeroed summary here would read as a measured 0 ms delta, so the
    # key is simply absent instead. Sibling of the aa_summary max() guard.
    if not R["rounds"]:
        print("summary: skipped, --rounds 0 (no fold A/B arms requested)", flush=True)
    else:
        R["summary"] = {
            "base_median_s": round(med(base), 4), "wire_median_s": round(med(wire), 4),
            "delta_ms_median": round((med(base) - med(wire)) * 1e3, 1),
            "delta_ms_paired_mean": round(sum(x["delta_ms"] for x in R["rounds"]) / len(R["rounds"]), 1),
            "delta_ms_min": min(x["delta_ms"] for x in R["rounds"]),
            "delta_ms_max": max(x["delta_ms"] for x in R["rounds"]),
            "n_positive_rounds": sum(1 for x in R["rounds"] if x["delta_ms"] > 0),
            "ratio": round(med(base) / med(wire), 5),
            "base_spread_ms": round((max(base) - min(base)) * 1e3, 1),
            "wire_spread_ms": round((max(wire) - min(wire)) * 1e3, 1),
            "bracketing_baselines_s": [base[0], base[len(base) // 2], base[-1]],
            "base_folds": base, "wire_folds": wire}
        print("summary:", R["summary"], flush=True)

    R["interaction_checks"] = {
        "l1_out_refused_after_all_folds": refused(),
        "trimul_throws": list(trimul_throws),
        "norm_l1_vs_dram_calls": dict(norm),
        "transpose_memory_config_before": R["transpose_memory_config_before"],
        "transpose_memory_config_after": transpose_branch(),
        "n_folds_total": 2 + 2 * a.aa_rounds + 2 * a.rounds}
    print("interaction:", R["interaction_checks"], flush=True)

    if len(coords) >= 2:
        c = coords
        R["parity_folds"] = {
            "n_coord_sets": len(c),
            "max_abs_delta_A_all_pairs": float(max(np.abs(c[i] - c[0]).max()
                                                   for i in range(1, len(c)))),
            "torch_equal_first_vs_last": bool(torch.equal(torch.as_tensor(c[0]),
                                                          torch.as_tensor(c[-1]))),
            "plddt_all": sorted({x["plddt_base"] for x in R["rounds"]}
                                | {x["plddt_wire"] for x in R["rounds"]}),
            "shape": list(c[0].shape)}
        print("parity_folds:", R["parity_folds"], flush=True)
    flush()

    # --- the block wall, the tighter instrument ----------------------------------------------------
    if "mod" in grab:
        T.TriangleMultiplication.__call__ = _orig_tm
        mod, xg, mg = grab["mod"], grab["x"], grab["mask"]

        def block(arm):
            RP.set_enabled(arm)
            n0 = RP.STATS[0]
            ts = []
            for i in range(a.block_reps + 3):
                x = ttnn.clone(xg, memory_config=xg.memory_config())
                m = ttnn.clone(mg, memory_config=mg.memory_config()) if mg is not None else None
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                y = mod(x, m)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) * 1e3
                ttnn.deallocate(y)
                if m is not None:
                    ttnn.deallocate(m)
                if i >= 3:
                    ts.append(dt)
            calls = (RP.STATS[0] - n0) / (a.block_reps + 3)
            RP.set_enabled(False)
            ts.sort()
            return ts[len(ts) // 2], calls

        bs, ws, cps = [], [], []
        for _ in range(3):
            b, _c = block(False); bs.append(b)
            w, c = block(True); ws.append(w); cps.append(c)
        d = med(bs) - med(ws)
        R["block_wall"] = {
            "shape": grab["shape"], "base_ms": [round(v, 4) for v in bs],
            "wire_ms": [round(v, 4) for v in ws],
            "base_median_ms": round(med(bs), 4), "wire_median_ms": round(med(ws), 4),
            "delta_ms_per_call": round(d, 4), "ratio": round(med(bs) / med(ws), 4),
            "eligible_calls_per_block": med(cps),
            "projected_ms_per_fold_x524x2": round(d * 2 * 524, 1),
            "bracket_base_first_last_ms": [round(bs[0], 4), round(bs[-1], 4)],
            "load": load()}
        print("block_wall:", R["block_wall"], flush=True)
    else:
        R["block_wall"] = {"error": "no TriangleMultiplication call captured"}
    flush()
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
