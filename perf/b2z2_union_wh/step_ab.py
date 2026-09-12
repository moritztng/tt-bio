#!/usr/bin/env python3
"""The three atom-branch levers on one settled diffusion step, together, interleaved on WH.

Six arms, all set as module globals so one process holds one device context and every arm sees
the same box:

    base     shipped
    G        TT_BIO_ATOM_KEY_WINDOW        the gather as a shape-gated window
    S        TT_BIO_ATOM_SHIFT_GATHER      the same gather, proved off the matrix instead
    L1       TT_BIO_ATOM_L1                the branch's live set in L1
    GK       G + TT_BIO_ATOM_KV_PREPROJ    the K/V projection moved inside the gather
    UNION    G + KV_PREPROJ + L1

`base` is timed at EVERY position in the block, so the A/A floor is the worst base-against-base
ratio in the same session rather than a number carried in from another one. The arm order is
reversed on alternate blocks, so a monotone drift in the box cancels instead of landing on
whichever arm runs last.

`--parity` compares every arm's step OUTPUT with torch.equal before any timing, with a negative
control that must differ.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))

#        arm  -> (KEY_WINDOW, KV_PREPROJ, ATOM_L1, SHIFT_GATHER)
ARMS = {
    "base":  (False, False, False, False),
    "G":     (True,  False, False, False),
    "S":     (False, False, False, True),
    "L1":    (False, False, True,  False),
    "GK":    (True,  True,  False, False),
    "SK":    (False, True,  False, True),
    "UNION": (True,  True,  True,  False),
    "SUNION": (False, True,  True,  True),
}


def set_arm(T, arm):
    T._ATOM_KEY_WINDOW, T._ATOM_KV_PREPROJ, T._ATOM_L1, T._ATOM_SHIFT_GATHER = ARMS[arm]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="base,G,S,L1,GK,SK,UNION,SUNION")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    arms = a.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm
    assert arms[0] == "base", "base must lead so the A/A floor has every position"
    for k in ("TT_BIO_ATOM_KEY_WINDOW", "TT_BIO_ATOM_KV_PREPROJ", "TT_BIO_ATOM_L1",
              "TT_BIO_ATOM_SHIFT_GATHER"):
        assert k not in os.environ, f"{k} is pinned in the env; the arms are set in process"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), \
        f"imported tt_bio from {_TB.__file__}, not this tree"
    import tt_bio.tenstorrent as T
    import step_probe as SP                    # inserts the paths tt_baseline is found on
    import tt_baseline as B

    SP.OUT_PATH = a.out.with_suffix(".grab.json")
    SP.OUT["env"] = {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename,
                   "tt_bio_file": _TB.__file__,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "arch": T.arch_name(),
                   "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                   "arms": arms, "reps": a.reps, "blocks": a.blocks,
                   "loadavg_start": open("/proc/loadavg").read().split()[:3]},
           "rows": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    out["env"]["grid"] = str(T.CORE_GRID_MAIN)
    ki = next((x for x in list(g["args"]) + list(g["kwargs"].values())
               if isinstance(x, T._AtomShiftGather)), None)
    out["env"]["shift_sentinel"] = ki is not None
    out["env"]["shift_windows"] = getattr(ki, "windows", None)

    def step():
        return g["obj"](*g["args"], **g["kwargs"])

    if a.parity and ki is not None:
        # Decisive: both constructions claim to BE the one-hot gather. Score each of the three
        # against the gather written out in float32 from the very matrix the fold cached, at the
        # production shape, so "bit-exact" is measured against the definition and not against
        # whichever arm happened to run first.
        m = ttnn.to_torch(ki.matrix).float()
        while m.dim() > 2:
            m = m[0]
        rows, cols = m.shape
        K, W, D = rows // 2, T.ATOM_WINDOW, T.ATOM_DIM
        src = torch.randn(1, K, W, D // 2, dtype=torch.bfloat16)
        ref = (src.reshape(1, 2 * K, W // 2, -1).permute(0, 2, 3, 1).float() @ m)
        ref = ref.permute(0, 3, 1, 2).reshape(1, K, -1, src.shape[3])
        st_ = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
        got = {}
        plan = T._atom_window_plan(st_.shape, ki.matrix)
        got["window"] = ttnn.to_torch(T._atom_window_gather(st_, plan)).float() if plan else None
        got["shift"] = ttnn.to_torch(T._atom_shift_gather(st_, ki.windows)).float()
        mm = ttnn.matmul(ttnn.permute(ttnn.reshape(st_, (1, 2 * K, W // 2, -1)), (0, 2, 3, 1)),
                         ki.matrix, core_grid=T.CORE_GRID_MAIN)
        got["onehot"] = ttnn.to_torch(
            ttnn.reshape(ttnn.permute(mm, (0, 3, 1, 2)), (1, K, -1, src.shape[3]))).float()
        gc = {"windows": int(ki.windows), "K": K, "windows_lt_K": int(ki.windows) < K,
              "matrix_shape": [rows, cols], "plan": plan is not None}
        for k, v in got.items():
            if v is None:
                gc[k] = None
                continue
            v = v[:, :, :ref.shape[2], :ref.shape[3]]
            gc[k] = {"bit_exact_vs_definition": bool(torch.equal(v, ref)),
                     "max_abs": float((v - ref).abs().max()),
                     "n_differing_rows": int((v - ref).abs().amax(dim=(0, 3)).gt(0).sum()),
                     "first_differing_window": (
                         int((v - ref).abs().amax(dim=(0, 2, 3)).gt(0).nonzero()[0])
                         if not torch.equal(v, ref) else None)}
        out["gather_check"] = gc
        print("GATHER_CHECK", json.dumps(gc, indent=1), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    if a.parity:
        ref, stats = {}, {}
        for arm in arms:
            set_arm(T, arm)
            T.ATOM_L1_STATS["l1"] = T.ATOM_L1_STATS["dram"] = 0
            T.ATOM_SHIFT_GATHER_STATS[0] = T.ATOM_SHIFT_GATHER_STATS[1] = 0
            ref[arm] = ttnn.to_torch(step())
            stats[arm] = {"atom_l1": dict(T.ATOM_L1_STATS),
                          "shift_gather": list(T.ATOM_SHIFT_GATHER_STATS)}
        par = {}
        for i, x in enumerate(arms):
            for y in arms[i + 1:]:
                par[f"{x}_vs_{y}"] = {
                    "bit_exact": bool(torch.equal(ref[x], ref[y])),
                    "max_abs": float((ref[x].float() - ref[y].float()).abs().max())}
        set_arm(T, arms[-1])
        again = ttnn.to_torch(step())
        par["self_repeat_bit_exact"] = bool(torch.equal(ref[arms[-1]], again))
        par["negative_control_differs"] = bool(not torch.equal(ref[arms[-1]],
                                                               ref[arms[0]] + 1.0))
        out["parity"], out["path_stats"] = par, stats
        print("PARITY", json.dumps(par, indent=1), flush=True)
        print("PATHS", json.dumps(stats), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    fence = SP.make_fence(ttnn, dev)
    for arm in arms:                                    # warm every program cache first
        set_arm(T, arm)
        for _ in range(3):
            step()
    ttnn.synchronize_device(dev)

    # base at every position: arms[0] leads, and one extra base is inserted after each other arm,
    # so the floor is measured against the same drift the ratios are.
    for blk in range(a.blocks):
        order = list(arms) if blk % 2 == 0 else list(reversed(arms))
        seq = []
        for arm in order:
            seq.append(arm)
            if arm != "base":
                seq.append("base")
        for pos, arm in enumerate(seq):
            set_arm(T, arm)
            fence()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                step()
            ttnn.synchronize_device(dev)
            ms = 1e3 * (time.perf_counter() - t0) / a.reps
            out["rows"].append({"block": blk, "pos": pos, "arm": arm, "ms": round(ms, 4),
                                "reversed": blk % 2 == 1,
                                "loadavg": open("/proc/loadavg").read().split()[0]})
            print(f"  blk{blk} p{pos:<2d} {arm:6s} {ms:8.4f} ms", flush=True)
            a.out.write_text(json.dumps(out, indent=1))

    def vals(arm):
        return [r["ms"] for r in out["rows"] if r["arm"] == arm]

    med = {arm: round(st.median(vals(arm)), 4) for arm in arms}
    base_by_pos = {}
    for r in out["rows"]:
        if r["arm"] == "base":
            base_by_pos.setdefault(r["pos"], []).append(r["ms"])
    pos_med = {p: st.median(v) for p, v in base_by_pos.items()}
    out["median_ms"] = med
    out["base_median_by_position"] = {str(p): round(v, 4) for p, v in sorted(pos_med.items())}
    out["aa_floor"] = round(max(pos_med.values()) / min(pos_med.values()), 5)
    out["spread_pct"] = {arm: round(100 * (max(vals(arm)) - min(vals(arm))) / med[arm], 3)
                         for arm in arms}
    out["ratio_vs_base"] = {arm: round(med["base"] / med[arm], 5) for arm in arms}
    out["arm_min_max"] = {arm: [round(min(vals(arm)), 4), round(max(vals(arm)), 4)]
                          for arm in arms}
    r = out["ratio_vs_base"]
    out["product_of_singles"], out["union_discount_pct"] = {}, {}
    for gather, kvp, union in (("G", "GK", "UNION"), ("S", "SK", "SUNION")):
        if all(k in r for k in (gather, "L1", kvp, union)):
            singles = r[gather] * r["L1"] * (r[kvp] / r[gather])
            out["product_of_singles"][union] = round(singles, 5)
            out["union_discount_pct"][union] = round(100 * (singles - r[union]) / singles, 3)
    if all(k in r for k in ("G", "L1", "GK", "UNION")):
        out["product_of_singles_this_session"] = out["product_of_singles"]["UNION"]
    if "S" in r and "G" in r:
        out["shift_vs_window"] = round(r["S"] / r["G"], 5)
    out["env"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.write_text(json.dumps(out, indent=1))
    for k in ("median_ms", "ratio_vs_base", "aa_floor", "product_of_singles",
              "union_discount_pct", "shift_vs_window", "arm_min_max"):
        if k in out:
            print(k.upper(), json.dumps(out[k]), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
