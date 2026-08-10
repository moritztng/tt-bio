#!/usr/bin/env python3
"""Deliverables 1b + 2 — per-model census and paired fold A/B for the `reblock_permute` flip.

Generalised from `perf/p3_permute_op/qb1_fold_ab.py` (X9, qb1) to any tt-bio fold model, because the
flip is a change to the SHARED `TriangleMultiplication` and the question is which of the other models
it touches. One process, model loaded once. Order:

  1. cold fold under each arm so no timed arm pays JIT compilation, and the census is taken on the
     cold ON fold: every `_channel_move` call with its shape, its in/out buffer type and the gate's
     verdict, plus the TriangleMultiplication invocation count per fold (the measured conversion
     from a block wall to a per-fold figure -- not a constant taken on trust);
  2. `--aa-rounds` A/A rounds, `reblock_permute` disabled in BOTH arms, so the only thing measured is
     this box's own resolution this session;
  3. `--rounds` A/B rounds, arms alternating, baseline restored in-session every round;
  4. the trimul block wall at the model's own pair shape, arms alternating, as the second instrument.

`ttnn.synchronize_device` on both sides of every timed region. Every figure is a qb2 / ttnn 0.68.0
RATIO and owes a qb1 re-take before it drives anything (charter §4.8).

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/crossmodel_ab.py \
        --model openfold3 --target examples/prot300.yaml --a3m scripts/gpu_vs_tt/fixtures/prot300.a3m
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--a3m", default="")
    ap.add_argument("--aa-rounds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--block-reps", type=int, default=11)
    ap.add_argument("--block-sets", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    label = a.label or f"{a.model}_{Path(a.target).stem}"
    out_path = Path(a.out) if a.out else OUT / f"ab_{label}.json"

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    target = REPO / a.target
    a3m = REPO / a.a3m if a.a3m else REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "ypx_msa"

    # ---- census instrumentation ------------------------------------------------------------------
    SEEN: dict = {}
    _orig_elig = RP.eligible

    def _elig(x, mc):
        v = _orig_elig(x, mc)
        key = (int(x.shape[1]), int(x.shape[3]),
               str(x.memory_config().buffer_type).rsplit(".", 1)[-1],
               str(mc.buffer_type).rsplit(".", 1)[-1],
               str(x.dtype).rsplit(".", 1)[-1], str(x.layout).rsplit(".", 1)[-1], bool(v))
        SEEN[key] = SEEN.get(key, 0) + 1
        return v

    RP.eligible = _elig

    TM_CALLS: dict = {}
    grab: dict = {"target": None}
    _orig_tm = T.TriangleMultiplication.__call__

    def _grab_tm(self, x, mask=None):
        s = tuple(int(d) for d in x.shape)
        TM_CALLS[s] = TM_CALLS.get(s, 0) + 1
        if "mod" not in grab and grab["target"] is not None and s == grab["target"]:
            grab["mod"] = self
            grab["x"] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            grab["mask"] = (ttnn.clone(mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            if mask is not None else None)
            grab["shape"] = list(s)
        return _orig_tm(self, x, mask)

    T.TriangleMultiplication.__call__ = _grab_tm

    from tt_baseline import build_fold
    t_load = time.perf_counter()
    one_fold, meta, state = build_fold(a.model, msa_dir, target, a3m)
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    coords: list = []
    if hasattr(state, "model") and hasattr(state.model, "fold"):
        _orig_fold = state.model.fold

        def _capture_fold(*ar, **kw):
            r = _orig_fold(*ar, **kw)
            c = r[0] if isinstance(r, tuple) else r
            try:
                coords.append(np.asarray(torch.as_tensor(c).detach().to(torch.float64).cpu()))
            except Exception:
                pass
            return r

        state.model.fold = _capture_fold

    struct_dir = Path(meta["struct_dir"])

    def cif_sha():
        h = []
        for p in sorted(struct_dir.glob("*")):
            if p.is_file():
                h.append((p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16]))
        return h

    import importlib.metadata as md
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    R = {"wheel": md.version("ttnn"), "host": "qb2",
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model,
         "target": a.target, "label": label,
         "grid": [g.x, g.y], "cores": g.x * g.y,
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
         "l1_max_seq": T._trimul_l1_max_seq(),
         "meta": {k: meta.get(k) for k in ("hardware", "card_type", "aiclk_mhz", "load_s", "n_msa",
                                          "timed_region", "diffusion_samples")},
         "aa": [], "rounds": []}

    # ---- 0. cold folds, and the census on the ON one ---------------------------------------------
    RP.set_enabled(False)
    c0, m0 = one_fold()
    sha_off = cif_sha()
    # The block wall must be taken at the pair shape the fold spends its trimul time on, so the
    # shape is chosen from the first fold's own invocation census rather than by a hardcoded guess.
    if TM_CALLS:
        grab["target"] = max(TM_CALLS.items(), key=lambda kv: (kv[1], kv[0][1]))[0]
        print("block-wall target shape:", grab["target"], "of", TM_CALLS, flush=True)
    SEEN.clear(); TM_CALLS.clear(); RP.STATS[0] = RP.STATS[1] = 0; RP.REJECTS.clear()
    RP.set_enabled(True)
    c1, m1 = one_fold()
    sha_on = cif_sha()
    RP.set_enabled(False)
    R["cold"] = {"base_s": round(c0, 3), "wire_s": round(c1, 3),
                 "n_tokens": m1.get("n_tokens"), "n_residues": m1.get("n_residues"),
                 "plddt_base": m0.get("plddt"), "plddt_wire": m1.get("plddt"),
                 "cif_sha_base": sha_off, "cif_sha_wire": sha_on,
                 "cif_sha_identical": sha_off == sha_on, "load": load()}
    R["census"] = {
        "channel_move_calls_per_fold": sum(SEEN.values()),
        "eligible_served_per_fold": RP.STATS[0],
        "refused_per_fold": RP.STATS[1],
        "by_shape": [{"N": k[0], "C": k[1], "in": k[2], "out": k[3], "dtype": k[4],
                      "layout": k[5], "eligible": k[6], "calls": v}
                     for k, v in sorted(SEEN.items())],
        "reject_reasons": {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()},
        "trimul_invocations_per_fold": {"x".join(str(d) for d in k): v
                                        for k, v in sorted(TM_CALLS.items())},
        "trimul_invocations_total": sum(TM_CALLS.values()),
    }
    print("cold:", json.dumps(R["cold"]), flush=True)
    print("census:", json.dumps(R["census"], indent=1), flush=True)
    out_path.write_text(json.dumps(R, indent=1))

    if R["census"]["eligible_served_per_fold"] == 0:
        R["verdict_shortcut"] = ("zero eligible calls on this input: the flip cannot change this "
                                 "model here, so no A/B is owed")
        print(R["verdict_shortcut"], flush=True)

    # ---- 1. the A/A control ----------------------------------------------------------------------
    for r in range(a.aa_rounds):
        RP.set_enabled(False)
        ta, _ = one_fold()
        RP.set_enabled(False)
        tb, _ = one_fold()
        row = {"round": r, "a_s": round(ta, 4), "b_s": round(tb, 4),
               "apparent_delta_ms": round((ta - tb) * 1e3, 1), "load": load()}
        R["aa"].append(row)
        print("aa:", row, flush=True)
        out_path.write_text(json.dumps(R, indent=1))

    if R["aa"]:
        aa_a = [x["a_s"] for x in R["aa"]]
        aa_b = [x["b_s"] for x in R["aa"]]
        allaa = aa_a + aa_b
        R["aa_summary"] = {
            "a_median_s": round(med(aa_a), 4), "b_median_s": round(med(aa_b), 4),
            "apparent_delta_ms_on_medians": round((med(aa_a) - med(aa_b)) * 1e3, 1),
            "max_abs_apparent_delta_ms": round(max(abs(x["apparent_delta_ms"]) for x in R["aa"]), 1),
            "all_folds_spread_ms": round((max(allaa) - min(allaa)) * 1e3, 1),
            "folds": [round(v, 4) for v in allaa]}
        print("aa_summary:", R["aa_summary"], flush=True)

    # ---- 2. the A/B ------------------------------------------------------------------------------
    for r in range(a.rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        tb, mb = one_fold()
        sb = cif_sha()
        n_base = RP.STATS[0] - n0
        RP.set_enabled(True)
        n1 = RP.STATS[0]
        tw, mw = one_fold()
        sw = cif_sha()
        n_wire = RP.STATS[0] - n1
        RP.set_enabled(False)
        row = {"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
               "delta_ms": round((tb - tw) * 1e3, 1),
               "eligible_served_base": n_base, "eligible_served_wire": n_wire,
               "plddt_base": mb.get("plddt"), "plddt_wire": mw.get("plddt"),
               "cif_sha_identical": sb == sw, "load": load()}
        R["rounds"].append(row)
        print("ab:", row, flush=True)
        out_path.write_text(json.dumps(R, indent=1))

    if R["rounds"]:
        base = [x["base_s"] for x in R["rounds"]]
        wire = [x["wire_s"] for x in R["rounds"]]
        R["summary"] = {
            "base_median_s": round(med(base), 4), "wire_median_s": round(med(wire), 4),
            "delta_ms_median": round((med(base) - med(wire)) * 1e3, 1),
            "delta_ms_paired_mean": round(sum(x["delta_ms"] for x in R["rounds"]) / len(R["rounds"]), 1),
            "delta_ms_min": min(x["delta_ms"] for x in R["rounds"]),
            "delta_ms_max": max(x["delta_ms"] for x in R["rounds"]),
            "signs": [1 if x["delta_ms"] > 0 else (-1 if x["delta_ms"] < 0 else 0) for x in R["rounds"]],
            "ratio": round(med(base) / med(wire), 5),
            "base_folds": base, "wire_folds": wire}
        print("summary:", R["summary"], flush=True)

    if len(coords) >= 2:
        c = coords
        R["parity"] = {
            "n_coord_sets": len(c),
            "max_abs_delta_all_pairs": float(max(np.abs(c[i] - c[0]).max() for i in range(1, len(c)))),
            "torch_equal_first_vs_last": bool(torch.equal(torch.as_tensor(c[0]), torch.as_tensor(c[-1]))),
            "shape": list(c[0].shape)}
        print("parity:", R["parity"], flush=True)

    # ---- 3. the block wall, the second instrument ------------------------------------------------
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
        for _ in range(a.block_sets):
            b, _c = block(False); bs.append(b)
            w, c = block(True); ws.append(w); cps.append(c)
        n_tm = sum(TM_CALLS.values()) or 1
        R["block_wall"] = {
            "shape": grab["shape"], "base_ms": [round(v, 4) for v in bs],
            "wire_ms": [round(v, 4) for v in ws],
            "base_median_ms": round(med(bs), 4), "wire_median_ms": round(med(ws), 4),
            "delta_ms_per_call": round(med(bs) - med(ws), 4),
            "ratio": round(med(bs) / med(ws), 4),
            "eligible_served_per_block": med(cps),
            "trimul_invocations_per_fold_measured": n_tm,
            "implied_delta_ms_per_fold": round((med(bs) - med(ws)) * n_tm, 1),
            "load": load()}
        print("block_wall:", json.dumps(R["block_wall"]), flush=True)

    RP.eligible = _orig_elig
    out_path.write_text(json.dumps(R, indent=1))
    print("wrote", out_path, flush=True)
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
