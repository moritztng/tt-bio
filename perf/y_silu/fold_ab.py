#!/usr/bin/env python3
"""y-silu -- the production A/B: Transition block wall + 298 aa fold wall, with the A/A floor first.

One process, model loaded once, `hoist=True` so a timed fold is `model.fold` only. Order, following
Z1 (`perf/p3_permute_op/qb1_fold_ab.py`):

  1. a cold fold under each arm, so no timed arm pays JIT compilation;
  2. `--aa-rounds` A/A rounds -- `_UNFUSED_SILU` False in both "arms", so the only thing measured is
     the fold wall's own resolution on this host at this load;
  3. `--rounds` A/B fold rounds, arms alternating, the baseline restored in-session every round;
  4. the Transition block wall at the fold's own pair shape [1, 298, 298, 256], arms alternating.
     This is the tighter instrument and the one the headline rests on: it prices a shared producer
     on the wall of the region it feeds rather than per call x calls.

Load average is recorded per fold; qb1 carries three sibling legs this pass and the fold wall is a
host-visible measurement.

    TT_VISIBLE_DEVICES=0 python3 perf/y_silu/fold_ab.py --aa-rounds 4 --rounds 4
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np, torch, ttnn


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa-rounds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--block-reps", type=int, default=9)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "fold_ab.json"))
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    res = dict(load_start=load(), out=a.out)

    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "w6_gate_msa"

    # capture one production Transition at the fold's own pair shape, for the block wall
    grab: dict = {}
    _orig_tr = T.Transition.__call__

    def _grab(self, x):
        if "inst" not in grab and len(x.shape) == 4 and int(x.shape[-1]) == 256:
            grab["inst"], grab["x"] = self, ttnn.to_torch(x).clone()
        return _orig_tr(self, x)

    T.Transition.__call__ = _grab

    from tt_baseline import build_fold
    t0 = time.perf_counter()
    one_fold, meta, state = build_fold("protenix-v2", msa_dir, target, a3m, hoist=True)
    print(f"model loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    dev = T.get_device()

    plddts: dict = {}
    coords: dict = {}
    _orig_fold = state.model.fold
    cur = {"arm": "cold"}

    def _cap(*ar, **kw):
        r = _orig_fold(*ar, **kw)
        c = r[0] if isinstance(r, tuple) else r
        coords.setdefault(cur["arm"], []).append(
            np.asarray(torch.as_tensor(c).detach().to(torch.float64).cpu()))
        return r

    state.model.fold = _cap

    def fold_once(arm):
        cur["arm"] = arm
        T._UNFUSED_SILU = (arm == "B")
        t = time.perf_counter()
        one_fold()
        return time.perf_counter() - t

    # 1. cold fold per arm (absorbs JIT for the unfused silu program)
    cold = {}
    for arm in ("A", "B"):
        cold[arm] = round(fold_once(arm), 3)
        print("cold", arm, cold[arm], load(), flush=True)
    res["cold_s"] = cold
    T.Transition.__call__ = _orig_tr

    # 2. A/A floor -- identical code in both "arms"
    aa = {"A1": [], "A2": []}
    for i in range(a.aa_rounds):
        for lbl in ("A1", "A2"):
            T._UNFUSED_SILU = False
            cur["arm"] = "aa"
            t = time.perf_counter(); one_fold(); dt = time.perf_counter() - t
            aa[lbl].append(round(dt, 4))
            print("AA", lbl, round(dt, 4), load(), flush=True)
    res["aa"] = aa
    res["aa_floor_ms"] = round(abs(med(aa["A1"]) - med(aa["A2"])) * 1e3, 1)
    res["aa_paired_mean_ms"] = round(
        float(np.mean([x - y for x, y in zip(aa["A1"], aa["A2"])])) * 1e3, 1)
    print("A/A floor ms:", res["aa_floor_ms"], "paired mean", res["aa_paired_mean_ms"], flush=True)

    # 3. A/B fold wall
    fw = {"A": [], "B": []}
    loads = []
    for i in range(a.rounds):
        for arm in ("A", "B"):
            dt = fold_once(arm)
            fw[arm].append(round(dt, 4))
            loads.append(load())
            print("AB", arm, round(dt, 4), load(), flush=True)
    T._UNFUSED_SILU = False
    res["fold_wall_s"] = fw
    res["fold_loads"] = loads
    res["fold_delta_ms_median"] = round((med(fw["A"]) - med(fw["B"])) * 1e3, 1)
    res["fold_delta_ms_paired"] = round(
        float(np.mean([x - y for x, y in zip(fw["A"], fw["B"])])) * 1e3, 1)
    res["fold_paired_positive"] = int(sum(1 for x, y in zip(fw["A"], fw["B"]) if x > y))
    print("fold delta ms (A-B) median", res["fold_delta_ms_median"],
          "paired", res["fold_delta_ms_paired"], flush=True)

    # 4. Transition block wall at the fold's own pair shape
    inst = grab["inst"]
    xz = ttnn.from_torch(grab["x"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res["block_shape"] = list(grab["x"].shape)

    def block(arm):
        T._UNFUSED_SILU = (arm == "B")
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        y = _orig_tr(inst, xz)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t
        ttnn.deallocate(y)
        return dt

    for arm in ("A", "B"):
        block(arm); block(arm)          # warm
    bw = {"A": [], "B": []}
    for i in range(a.block_reps):
        for arm in ("A", "B"):
            bw[arm].append(round(block(arm) * 1e6, 2))
    T._UNFUSED_SILU = False
    res["block_wall_us"] = bw
    res["block_med_us"] = {k: round(med(v), 2) for k, v in bw.items()}
    res["block_delta_us"] = round(med(bw["A"]) - med(bw["B"]), 2)
    print("block wall us", res["block_med_us"], "delta", res["block_delta_us"], flush=True)

    # per-fold conversion: transition_z executions at c_z=256 counted in the census fold
    res["transition_z_cz256_calls_per_fold"] = 524
    res["block_delta_ms_per_fold"] = round(res["block_delta_us"] * 524 / 1e3, 1)
    print("block wall ms/fold:", res["block_delta_ms_per_fold"], flush=True)

    # 5. structural parity on the fold output
    par = {}
    try:
        ca = np.stack(coords["A"]); cb = np.stack(coords["B"])
        par["n_folds"] = [len(coords["A"]), len(coords["B"])]
        par["coord_max_abs"] = float(np.abs(ca[-1] - cb[-1]).max())
        par["coord_rmsd"] = float(np.sqrt(((ca[-1] - cb[-1]) ** 2).sum(-1).mean()))
        par["A_self_rmsd"] = float(np.sqrt(((ca[0] - ca[-1]) ** 2).sum(-1).mean()))
    except Exception as e:
        par["error"] = repr(e)[:300]
    res["fold_coord_parity"] = par
    res["load_end"] = load()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
