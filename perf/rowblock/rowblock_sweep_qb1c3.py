#!/usr/bin/env python3
"""z-rowblock exec pass: settle (R, group) on the real production helper, and prove parity.

Every cell here calls `tt_bio.tenstorrent._pair_transpose` itself -- the code the fold runs -- with
`_TRANSPOSE_ROWBLOCK_R` / `_TRANSPOSE_ROWBLOCK_GROUP` forced, so the sweep and the parity check are
statements about production and not about a probe that resembles it. `group == ceil(S/R)` is the
sibling's all-blocks-live form (probe variant C) and `group == 1` is the one-block-live form (D);
the plan's default rule is measured against both.

Roofs are re-measured on this card in this process (charter 4.1) and every timed region is synced
on both sides (`ttnn-sync-before-every-timed-region`).
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch   # noqa: E402
import ttnn    # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=1, pipe=3, reps=7):
    for _ in range(warm):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            o = fn()
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out), round((max(out) - min(out)) / st.median(out) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="416,512,640,1095")
    ap.add_argument("--parity-sizes", default="416,512")
    ap.add_argument("--blocks", default="32,64,96,128,192,256")
    ap.add_argument("--groups", default="1,2,4,8,16,all")
    a = ap.parse_args()

    import importlib.metadata as im
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    budget = per_core * gx * gy
    res = {"host": "qb1", "card_visible": "3", "ttnn": im.version("ttnn"),
           "core_grid_main": f"{gx}x{gy}", "l1_per_core_bytes": per_core,
           "l1_budget_bytes": budget,
           "reblock_permute_env": "default 0 (off) -- z-permute-bands owns that flag",
           "runs": []}
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)

    def emit(rec):
        res["runs"].append(rec)
        print("  " + json.dumps(rec), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))

    par_sizes = {int(v) for v in a.parity_sizes.split(",")}
    Rs = [int(v) for v in a.blocks.split(",")]
    groups = a.groups.split(",")

    # ---- roofs on this card, this pass, at the 512 aa pair shape ----------------------------
    xt = torch.randn(512, 512, 256, dtype=torch.float32).to(torch.bfloat16)
    x512 = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=DRAM)
    mb = 512 * 512 * 256 * 2 / 1e6
    for rn, mc in (("copy roof (DRAM), DRAM source", DRAM), ("copy roof (L1), DRAM source", L1)):
        t, sp = timed(dev, lambda mc=mc: ttnn.clone(x512, memory_config=mc))
        emit({"variant": "roof", "roof": rn, "shape": [512, 512, 256], "ms": round(t * 1e3, 4),
              "moved_MB": round(2 * mb, 2), "GB_s_read_plus_write": round(2 * mb / 1e3 / t, 1),
              "spread_pct_of_median": sp})
    xl1 = ttnn.clone(x512, memory_config=L1)
    t, sp = timed(dev, lambda: ttnn.clone(xl1, memory_config=DRAM))
    emit({"variant": "roof", "roof": "copy roof (DRAM), L1 source", "shape": [512, 512, 256],
          "ms": round(t * 1e3, 4), "GB_s_read_plus_write": round(2 * mb / 1e3 / t, 1),
          "spread_pct_of_median": sp})
    ttnn.deallocate(xl1)
    ttnn.deallocate(x512)

    # ---- the (R, group) sweep, on the production helper -------------------------------------
    for N in [int(v) for v in a.sizes.split(",")]:
        S = -(-N // 32) * 32
        xt = torch.randn(N, S, 256, dtype=torch.float32).to(torch.bfloat16)
        ref = xt.permute(1, 0, 2).contiguous() if N in par_sizes else None
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=DRAM)
        full_mb = N * S * 256 * 2 / 1e6
        fit = T._transpose_memory_config(x)
        emit({"N": N, "S": S, "shape": [N, S, 256], "full_MB": round(full_mb, 2),
              "production_fit_decision": "L1" if fit.buffer_type == ttnn.BufferType.L1 else "DRAM",
              "default_plan_R_group": list(T._rowblock_plan(x) or []),
              "note": "the branch actually taken, read live from the helper"})

        def check(rec, mk):
            if ref is None:
                return rec
            try:
                o = mk()
                got = ttnn.to_torch(o)
                ttnn.deallocate(o)
                rec["torch_equal"] = bool(torch.equal(got, ref))
                rec["max_abs_diff"] = float((got.float() - ref.float()).abs().max())
                rec["shape_out"] = list(got.shape)
            except Exception as e:                                             # noqa: BLE001
                rec["parity_error"] = f"{type(e).__name__}: {e}"[:200]
            return rec

        # A: production today -- the unblocked permute at whatever the fit test says
        T._TRANSPOSE_ROWBLOCK = False
        mkA = lambda: T._pair_transpose(x)                                     # noqa: E731
        recA = {"N": N, "variant": "A_unblocked", "peak_l1_MB": 0.0,
                "dest": "L1" if fit.buffer_type == ttnn.BufferType.L1 else "DRAM"}
        try:
            t, sp = timed(dev, mkA)
            recA["ms"], recA["spread_pct_of_median"] = round(t * 1e3, 4), sp
        except Exception as e:                                                 # noqa: BLE001
            recA["error"] = f"{type(e).__name__}: {e}"[:200]
        emit(check(recA, mkA))
        baseA = recA.get("ms")
        T._TRANSPOSE_ROWBLOCK = True

        for R in Rs:
            if R >= S:
                continue
            nblk = -(-S // R)
            blk = T._rowblock_bytes(x, R)
            for g in groups:
                gi = nblk if g == "all" else int(g)
                if gi > nblk or (g != "all" and gi == nblk):
                    continue          # 'all' already covers group == nblk
                T._TRANSPOSE_ROWBLOCK_R, T._TRANSPOSE_ROWBLOCK_GROUP = R, gi
                mk = lambda: T._pair_transpose(x)                              # noqa: E731
                rec = {"N": N, "variant": "blocked", "R": R, "group": gi, "blocks": nblk,
                       "group_label": g, "block_MB": round(blk / 1e6, 2),
                       "peak_l1_MB": round(min(gi, nblk) * blk / 1e6, 2),
                       "ragged_last_block": S % R != 0, "last_block_rows": S - (nblk - 1) * R}
                try:
                    t, sp = timed(dev, mk)
                    rec["ms"], rec["spread_pct_of_median"] = round(t * 1e3, 4), sp
                    if baseA:
                        rec["vs_A"] = round(baseA / rec["ms"], 3)
                        rec["ms_saved_per_call"] = round(baseA - rec["ms"], 4)
                except Exception as e:                                         # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:200]
                emit(check(rec, mk))
        T._TRANSPOSE_ROWBLOCK_R = T._TRANSPOSE_ROWBLOCK_GROUP = 0
        ttnn.deallocate(x)

    # ---- the three named parity shapes, at the default plan ---------------------------------
    for shape in ([512, 512, 256], [298, 320, 256], [512, 507, 256], [320, 298, 256]):
        d0, d1, c = shape
        xt = torch.randn(d0, d1, c, dtype=torch.float32).to(torch.bfloat16)
        ref = xt.permute(1, 0, 2).contiguous()
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=DRAM)
        rec = {"variant": "parity_at_fold_shape", "shape": shape,
               "plan": list(T._rowblock_plan(x) or []),
               "production_fit_decision":
                   "L1" if T._transpose_memory_config(x).buffer_type == ttnn.BufferType.L1
                   else "DRAM"}
        for force in (False, True):
            T._TRANSPOSE_ROWBLOCK = force
            try:
                o = T._pair_transpose(x)
                got = ttnn.to_torch(o)
                ttnn.deallocate(o)
                rec[f"rowblock={force}"] = {
                    "torch_equal": bool(torch.equal(got, ref)),
                    "max_abs_diff": float((got.float() - ref.float()).abs().max()),
                    "shape_out": list(got.shape)}
            except Exception as e:                                             # noqa: BLE001
                rec[f"rowblock={force}"] = {"error": f"{type(e).__name__}: {e}"[:250]}
        T._TRANSPOSE_ROWBLOCK = True
        emit(rec)
        ttnn.deallocate(x)

    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
