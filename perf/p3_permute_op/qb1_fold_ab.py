#!/usr/bin/env python3
"""Deliverable 2: the noise floor first, then the campaign absolute, in one session on qb1 card 0.

X6's 188.8 ms/fold is a qb2 / ttnn 0.68.0 ratio. This is the qb1 / 0.67.4 re-take, and it takes the
A/A control BEFORE the A/B so the delta is scored against a floor measured on this card rather than
against X6's.

One process, model loaded once, `hoist=True` so the timed region is `model.fold` only. Order:

  1. cold fold under each arm, so no timed arm pays JIT compilation;
  2. `--aa-rounds` A/A rounds -- identical code in both "arms", `reblock_permute` disabled in both,
     so the only thing measured is the harness's own resolution;
  3. `--rounds` A/B rounds, arms alternating, baseline restored in-session every round;
  4. the trimul block wall at the fold's own shape, arms alternating, as the second instrument.

Host load average is recorded per fold: qb1's other cards carry sibling legs this pass and the fold
wall is a host-visible measurement.

    TT_VISIBLE_DEVICES=0 python3 perf/p3_permute_op/qb1_fold_ab.py --aa-rounds 5 --rounds 5
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


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa-rounds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--block-reps", type=int, default=15)
    ap.add_argument("--out", default=str(OUT / "qb1_fold_ab.json"))
    ap.add_argument("--card", type=int, default=0, help="label only; the card comes from TT_VISIBLE_DEVICES")
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "w6_gate_msa"

    from tt_baseline import build_fold
    t_load = time.perf_counter()
    one_fold, meta, state = build_fold("protenix-v2", msa_dir, target, a3m, hoist=True)
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    coords: list = []
    _orig_fold = state.model.fold

    def _capture_fold(*ar, **kw):
        r = _orig_fold(*ar, **kw)
        c = r[0] if isinstance(r, tuple) else r
        coords.append(np.asarray(torch.as_tensor(c).detach().to(torch.float64).cpu()))
        return r

    state.model.fold = _capture_fold

    # One production TriangleMultiplication at the fold's own pair shape, for the block wall.
    grab: dict = {}
    _orig_tm = T.TriangleMultiplication.__call__

    def _grab_tm(self, x, mask=None):
        if "mod" not in grab and int(x.shape[1]) == int(x.shape[2]) and int(x.shape[3]) >= 128:
            grab["mod"] = self
            grab["x"] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            grab["mask"] = ttnn.clone(mask, memory_config=ttnn.DRAM_MEMORY_CONFIG) if mask is not None else None
            grab["shape"] = [int(d) for d in x.shape]
        return _orig_tm(self, x, mask)

    T.TriangleMultiplication.__call__ = _grab_tm

    R = {"wheel": "0.67.4", "host": "qb1", "card": a.card,
         "meta": {"hardware": meta["hardware"], "card_type": meta.get("card_type"),
                  "aiclk_mhz": meta.get("aiclk_mhz"), "load_s": meta["load_s"],
                  "n_msa": meta["n_msa"]},
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
         "aa": [], "rounds": []}

    def transpose_branch():
        t = ttnn.from_torch(torch.zeros(298, 320, 256, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            device=T.get_device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = str(T._transpose_memory_config(t).buffer_type)
        ttnn.deallocate(t)
        return b

    R["transpose_memory_config_before"] = transpose_branch()

    # `_l1_layer_norm` returns (tensor, in_l1); count both branches over the whole session so a
    # silent DRAM fallback under the kernel's circular buffers cannot hide (charter §4.10).
    norm_counts = [0, 0]
    _orig_norm = T._l1_layer_norm

    def _counting_norm(x, headroom, **kw):
        r = _orig_norm(x, headroom, **kw)
        norm_counts[0 if r[1] else 1] += 1
        return r

    T._l1_layer_norm = _counting_norm

    def free_l1():
        try:
            v = ttnn.get_memory_view(T.get_device(), ttnn.BufferType.L1)
            return {"largest_contiguous_bytes_free_per_bank":
                        int(v.largest_contiguous_bytes_free_per_bank),
                    "total_bytes_free_per_bank": int(v.total_bytes_free_per_bank),
                    "total_bytes_per_bank": int(v.total_bytes_per_bank)}
        except Exception as e:                                                 # noqa: BLE001
            return {"error": repr(e)[:200]}

    R["free_l1_idle"] = free_l1()

    RP.set_enabled(False)
    c0, m0 = one_fold()
    RP.set_enabled(True)
    c1, m1 = one_fold()
    RP.set_enabled(False)
    R["cold"] = {"base_s": round(c0, 3), "wire_s": round(c1, 3), "n_tokens": m1.get("n_tokens"),
                 "plddt": m1.get("plddt"), "eligible_calls_in_one_wire_fold": RP.STATS[0],
                 "load": load()}
    print("cold:", R["cold"], flush=True)
    R["transpose_memory_config_after"] = transpose_branch()

    # ---- 1. the A/A control: identical code in both arms ----------------------------------------
    for r in range(a.aa_rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        ta, _ = one_fold()
        RP.set_enabled(False)          # the "other" arm is the SAME arm, on purpose
        tb, _ = one_fold()
        row = {"round": r, "a_s": round(ta, 4), "b_s": round(tb, 4),
               "apparent_delta_ms": round((ta - tb) * 1e3, 1),
               "eligible_calls": RP.STATS[0] - n0, "load": load()}
        R["aa"].append(row)
        print("aa:", row, flush=True)

    aa_a = [x["a_s"] for x in R["aa"]]
    aa_b = [x["b_s"] for x in R["aa"]]
    allaa = aa_a + aa_b
    R["aa_summary"] = {
        "a_median_s": round(med(aa_a), 4), "b_median_s": round(med(aa_b), 4),
        "apparent_delta_ms_on_medians": round((med(aa_a) - med(aa_b)) * 1e3, 1),
        "max_abs_apparent_delta_ms": round(max(abs(x["apparent_delta_ms"]) for x in R["aa"]), 1),
        "a_spread_ms": round((max(aa_a) - min(aa_a)) * 1e3, 1),
        "b_spread_ms": round((max(aa_b) - min(aa_b)) * 1e3, 1),
        "all_folds_spread_ms": round((max(allaa) - min(allaa)) * 1e3, 1),
        "folds": [round(v, 4) for v in allaa],
    }
    print("aa_summary:", R["aa_summary"], flush=True)

    # ---- 2. the A/B ------------------------------------------------------------------------------
    for r in range(a.rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        tb, mb = one_fold()
        n_base = RP.STATS[0] - n0
        RP.set_enabled(True)
        n1 = RP.STATS[0]
        tw, mw = one_fold()
        n_wire = RP.STATS[0] - n1
        RP.set_enabled(False)
        row = {"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
               "delta_ms": round((tb - tw) * 1e3, 1),
               "eligible_calls_base": n_base, "eligible_calls_wire": n_wire,
               "plddt_base": mb.get("plddt"), "plddt_wire": mw.get("plddt"), "load": load()}
        R["rounds"].append(row)
        print("ab:", row, flush=True)

    base = [x["base_s"] for x in R["rounds"]]
    wire = [x["wire_s"] for x in R["rounds"]]
    R["rejects"] = {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()}
    R["summary"] = {
        "base_median_s": round(med(base), 4), "wire_median_s": round(med(wire), 4),
        "delta_ms_median": round((med(base) - med(wire)) * 1e3, 1),
        "delta_ms_paired_mean": round(sum(x["delta_ms"] for x in R["rounds"]) / len(R["rounds"]), 1),
        "delta_ms_min": min(x["delta_ms"] for x in R["rounds"]),
        "delta_ms_max": max(x["delta_ms"] for x in R["rounds"]),
        "ratio": round(med(base) / med(wire), 5),
        "base_spread_ms": round((max(base) - min(base)) * 1e3, 1),
        "wire_spread_ms": round((max(wire) - min(wire)) * 1e3, 1),
        "base_folds": base, "wire_folds": wire,
    }
    print("summary:", R["summary"], flush=True)

    if len(coords) >= 2:
        c = coords
        R["parity"] = {
            "n_coord_sets": len(c),
            "max_abs_delta_A_all_pairs": float(max(np.abs(c[i] - c[0]).max() for i in range(1, len(c)))),
            "torch_equal_first_vs_last": bool(torch.equal(torch.as_tensor(c[0]), torch.as_tensor(c[-1]))),
            "shape": list(c[0].shape),
        }
        print("parity:", R["parity"], flush=True)

    # ---- 3. the block wall, the second instrument ------------------------------------------------
    if "mod" in grab:
        T.TriangleMultiplication.__call__ = _orig_tm
        dev = T.get_device()
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
        R["block_wall"] = {
            "shape": grab["shape"], "base_ms": [round(v, 4) for v in bs],
            "wire_ms": [round(v, 4) for v in ws],
            "base_median_ms": round(med(bs), 4), "wire_median_ms": round(med(ws), 4),
            "delta_ms_per_call": round(med(bs) - med(ws), 4),
            "ratio": round(med(bs) / med(ws), 4),
            "eligible_calls_per_block": med(cps),
            "load": load(),
        }
        # charter §4.9: x524 c_z=256 PairformerLayer executions, 2 trimuls each.
        d = R["block_wall"]["delta_ms_per_call"]
        R["block_wall"]["projected_ms_per_fold_x524x2"] = round(d * 2 * 524, 1)
        print("block_wall:", R["block_wall"], flush=True)
    else:
        R["block_wall"] = {"error": "no TriangleMultiplication call captured"}

    R["interaction"] = {
        "l1_layer_norm_l1_branch": norm_counts[0],
        "l1_layer_norm_dram_fallback": norm_counts[1],
        "L1_OUT_REFUSED": sorted(str(k) for k in T._L1_OUT_REFUSED),
        "transpose_memory_config_before": R["transpose_memory_config_before"],
        "transpose_memory_config_after": R["transpose_memory_config_after"],
        "free_l1_idle": R["free_l1_idle"],
        "free_l1_end": free_l1(),
        "trimul_cb_throw": False,
    }
    print("interaction:", R["interaction"], flush=True)

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
