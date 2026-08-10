#!/usr/bin/env python3
"""Deliverable 2: the production A/B, arms alternating in one process with the model loaded once.

BASE is the unmodified `ttnn.permute` path; WIRE routes the `(0,3,1,2)` chunk move through the
custom kernel behind the measured gate. The two arms differ only by `reblock_permute.set_enabled`,
so there is no worktree edit between arms and the baseline is restored in-session on every round.

Reports, per round: the whole-fold wall (hoisted region = `model.fold` only), the eligible-call
count the gate actually served, and the final coordinates for the parity verdict. Then, after the
folds, replays the captured production TriangleMultiplication at the fold's own shape under both
arms for the block wall, which is the instrument with the resolution the fold wall may lack.

    python3 perf/p3_permute_op/wire_fold_ab.py --rounds 5
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np
import torch
import ttnn

OUT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--block-reps", type=int, default=15)
    ap.add_argument("--out", default=str(OUT / "wire_fold_ab.json"))
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

    # Capture the coordinates the hoisted region produces (it suppresses the CIF write).
    coords: list = []
    _orig_fold = state.model.fold

    def _capture_fold(*ar, **kw):
        r = _orig_fold(*ar, **kw)
        c = r[0] if isinstance(r, tuple) else r
        coords.append(np.asarray(torch.as_tensor(c).detach().to(torch.float64).cpu()))
        return r

    state.model.fold = _capture_fold

    # Capture one production TriangleMultiplication call at the fold's own shape for the block wall.
    grab: dict = {}
    _orig_tm = T.TriangleMultiplication.__call__

    def _grab_tm(self, x, mask=None):
        if "mod" not in grab and int(x.shape[1]) == int(x.shape[2]):
            grab["mod"] = self
            grab["x"] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            grab["mask"] = ttnn.clone(mask, memory_config=ttnn.DRAM_MEMORY_CONFIG) if mask is not None else None
            grab["shape"] = [int(d) for d in x.shape]
        return _orig_tm(self, x, mask)

    T.TriangleMultiplication.__call__ = _grab_tm

    R = {"rounds": [], "meta": {"hardware": meta["hardware"], "card_type": meta.get("card_type"),
                                "aiclk_mhz": meta.get("aiclk_mhz"), "load_s": meta["load_s"],
                                "n_msa": meta["n_msa"]}}

    # W8 / clause 6: which branch does W6's landed lever take, at the fold's own pair shape?
    def transpose_branch():
        t = ttnn.from_torch(torch.zeros(298, 320, 256, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            device=T.get_device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = str(T._transpose_memory_config(t).buffer_type)
        ttnn.deallocate(t)
        return b

    R["transpose_memory_config_before"] = transpose_branch()

    # Cold folds: one per arm, so neither timed arm pays JIT compilation or first-touch.
    RP.set_enabled(False)
    c0, m0 = one_fold()
    RP.set_enabled(True)
    c1, m1 = one_fold()
    RP.set_enabled(False)
    R["cold"] = {"base_s": round(c0, 3), "wire_s": round(c1, 3),
                 "n_tokens": m1.get("n_tokens"), "plddt": m1.get("plddt"),
                 "eligible_calls_in_one_wire_fold": RP.STATS[0]}
    print(R["cold"], flush=True)
    R["transpose_memory_config_after"] = transpose_branch()

    for r in range(a.rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        tb, mb = one_fold()
        n_base = RP.STATS[0] - n0
        RP.set_enabled(True)
        n1 = RP.STATS[0]
        tw, mw = one_fold()
        n_wire = RP.STATS[0] - n1
        RP.set_enabled(False)  # baseline restored in-session, every round
        row = {"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
               "delta_ms": round((tb - tw) * 1e3, 1),
               "eligible_calls_base": n_base, "eligible_calls_wire": n_wire,
               "plddt_base": mb.get("plddt"), "plddt_wire": mw.get("plddt")}
        R["rounds"].append(row)
        print(row, flush=True)

    base = [x["base_s"] for x in R["rounds"]]
    wire = [x["wire_s"] for x in R["rounds"]]
    med = lambda v: sorted(v)[len(v) // 2]
    R["summary"] = {
        "base_median_s": round(med(base), 4), "wire_median_s": round(med(wire), 4),
        "delta_ms_median": round((med(base) - med(wire)) * 1e3, 1),
        "ratio": round(med(base) / med(wire), 5),
        "base_spread_ms": round((max(base) - min(base)) * 1e3, 1),
        "wire_spread_ms": round((max(wire) - min(wire)) * 1e3, 1),
        "base_folds": base, "wire_folds": wire,
    }
    print(R["summary"], flush=True)

    # Parity: every fold in this session used the same seed, so all coordinate sets must agree.
    if len(coords) >= 2:
        c = coords
        R["parity"] = {
            "n_coord_sets": len(c),
            "max_abs_delta_A_all_pairs": float(max(np.abs(c[i] - c[0]).max() for i in range(1, len(c)))),
            "torch_equal_first_vs_last": bool(torch.equal(torch.as_tensor(c[0]), torch.as_tensor(c[-1]))),
            "shape": list(c[0].shape),
        }
        print(R["parity"], flush=True)

    # ---- the block wall, at the fold's own shape -----------------------------------------------
    if "mod" in grab:
        T.TriangleMultiplication.__call__ = _orig_tm
        dev = T.get_device()
        mod, xg, mg = grab["mod"], grab["x"], grab["mask"]

        def block(arm):
            RP.set_enabled(arm)
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
            RP.set_enabled(False)
            ts.sort()
            return ts[len(ts) // 2], ts

        # Alternate the arms rather than running one block of each.
        bs, ws = [], []
        for _ in range(3):
            b, _all = block(False); bs.append(b)
            w, _all = block(True); ws.append(w)
        R["block_wall"] = {
            "shape": grab["shape"], "ending": bool(getattr(mod, "ending", None)),
            "base_ms": [round(v, 4) for v in bs], "wire_ms": [round(v, 4) for v in ws],
            "base_median_ms": round(med(bs), 4), "wire_median_ms": round(med(ws), 4),
            "delta_ms_per_call": round(med(bs) - med(ws), 4),
            "ratio": round(med(bs) / med(ws), 4),
            "calls_per_fold_x524x2": None,
        }
        print(R["block_wall"], flush=True)
    else:
        R["block_wall"] = {"error": "no TriangleMultiplication call captured"}

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
