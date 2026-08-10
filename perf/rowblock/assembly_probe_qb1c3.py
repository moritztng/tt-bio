#!/usr/bin/env python3
"""z-rowblock, planning pass: what does a PRODUCTION-LEGAL blocked transpose cost?

The sibling `size512-ab` priced row-blocking at 1861.9 ms/fold at 512 aa. Its probe
(`perf/size512/blocked_permute_qb2c0.py`) slices a column strip, permutes it to L1, appends the
result to a list and deallocates the whole list at the end. It therefore measures a form that

  (a) never assembles the blocks back into the single tensor the two production call sites return, and
  (b) holds every block in L1 at once -- 134.2 MB of qb2's 168.6 MB budget at N=512 -- which is the
      opposite of the size-independence the blocking is for. At N=1095 the same form needs 628 MB.

This probe prices the assembly and separates the two properties. Every variant produces the SAME
full [S, N, 256] DRAM tensor production needs, and every variant is checked with torch.equal.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=1, pipe=2, reps=3):
    for _ in range(warm):
        o = fn()
        if o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            o = fn()
            if o is not None:
                ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="298,384,512")
    ap.add_argument("--blocks", default="32,64,128,256")
    a = ap.parse_args()

    import importlib.metadata as im
    # Read the grid THROUGH the module: `COMPUTE_GRID_MAIN` is rewritten at device open
    # (tenstorrent.py:966-979), so `from tt_bio.tenstorrent import COMPUTE_GRID_MAIN` captures the
    # pre-open default (11, 10) and every L1 budget derived from it comes out 15.4 % low on qb1.
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device, _transpose_memory_config

    dev = get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    budget = per_core * gx * gy
    res = {"host": "qb1", "card_visible": "3", "ttnn": im.version("ttnn"),
           "core_grid_main": f"{gx}x{gy}", "l1_per_core_bytes": per_core,
           "l1_budget_bytes": budget, "runs": []}
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)

    def emit(rec):
        res["runs"].append(rec)
        print("  " + json.dumps(rec), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))

    for N in [int(v) for v in a.sizes.split(",")]:
        S = -(-N // 32) * 32
        xt = torch.randn(N, S, 256, dtype=torch.float32).to(torch.bfloat16)
        ref = xt.permute(1, 0, 2).contiguous()
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        full_mb = N * S * 256 * 2 / 1e6
        prod_dec = "L1" if _transpose_memory_config(x) is L1 else "DRAM"
        emit({"N": N, "S": S, "shape": [N, S, 256], "full_MB": round(full_mb, 2),
              "production_fit_decision": prod_dec,
              "note": "production _transpose_memory_config decision at this shape"})

        def check(mk, name, rec):
            """Run once outside the timed loop and torch.equal it."""
            try:
                o = mk()
                got = ttnn.to_torch(o)
                ttnn.deallocate(o)
                rec["torch_equal"] = bool(torch.equal(got, ref))
                rec["shape_out"] = list(got.shape)
            except Exception as e:                                            # noqa: BLE001
                rec["parity_error"] = str(e)[:200]
            return rec

        # --- roofs on this card, this pass -----------------------------------------------------
        for rn, mc in (("clone_dram_to_dram", DRAM), ("clone_dram_to_l1", L1)):
            rec = {"N": N, "variant": "roof", "roof": rn}
            try:
                t = timed(dev, lambda mc=mc: ttnn.clone(x, memory_config=mc))
                rec["ms"] = round(t * 1e3, 4)
                rec["moved_MB"] = round(2 * full_mb, 2)
                rec["GB_s_two_way"] = round(2 * full_mb / 1e3 / t, 1)
            except Exception as e:                                            # noqa: BLE001
                rec["error"] = str(e)[:200]
            emit(rec)
        xl1 = None
        try:
            xl1 = ttnn.clone(x, memory_config=L1)
            rec = {"N": N, "variant": "roof", "roof": "clone_l1_to_dram"}
            t = timed(dev, lambda: ttnn.clone(xl1, memory_config=DRAM))
            rec["ms"] = round(t * 1e3, 4)
            rec["GB_s_two_way"] = round(2 * full_mb / 1e3 / t, 1)
            emit(rec)
        except Exception as e:                                                # noqa: BLE001
            emit({"N": N, "variant": "roof", "roof": "clone_l1_to_dram", "error": str(e)[:200]})
        finally:
            if xl1 is not None:
                ttnn.deallocate(xl1)

        # --- A: production today ---------------------------------------------------------------
        mkA = lambda: ttnn.permute(x, (1, 0, 2), memory_config=DRAM)
        recA = {"N": N, "variant": "A_full_dram", "calls": 1, "peak_l1_MB": 0.0}
        try:
            recA["ms"] = round(timed(dev, mkA) * 1e3, 4)
        except Exception as e:                                                # noqa: BLE001
            recA["error"] = str(e)[:200]
        emit(check(mkA, "A", recA))
        baseA = recA.get("ms")

        # --- B: full permute to L1, then one contiguous copy out -------------------------------
        def mkB():
            o = ttnn.permute(x, (1, 0, 2), memory_config=L1)
            d = ttnn.to_memory_config(o, DRAM)
            ttnn.deallocate(o)
            return d
        recB = {"N": N, "variant": "B_full_l1_then_copy", "calls": 2,
                "peak_l1_MB": round(full_mb, 2),
                "l1_headroom_available": round(budget / (full_mb * 1e6), 3)}
        try:
            recB["ms"] = round(timed(dev, mkB) * 1e3, 4)
            if baseA:
                recB["vs_A"] = round(baseA / recB["ms"], 3)
        except Exception as e:                                                # noqa: BLE001
            recB["error"] = str(e)[:200]
        emit(check(mkB, "B", recB))

        for R in [int(v) for v in a.blocks.split(",")]:
            if R > S:
                continue
            nblk = -(-S // R)
            blk_mb = N * R * 256 * 2 / 1e6

            # --- C: all blocks live in L1, one concat out (the sibling's residency, assembled) --
            def mkC(R=R):
                outs = []
                for s in range(0, S, R):
                    b = x[:, s:min(s + R, S), :]
                    outs.append(ttnn.permute(b, (1, 0, 2), memory_config=L1))
                    ttnn.deallocate(b)
                o = ttnn.concat(outs, dim=0, memory_config=DRAM)
                for t_ in outs:
                    ttnn.deallocate(t_)
                return o
            recC = {"N": N, "variant": "C_blocks_l1_concat", "R": R, "calls": 2 * nblk + 1,
                    "blocks": nblk, "block_MB": round(blk_mb, 2),
                    "peak_l1_MB": round(nblk * blk_mb, 2),
                    "block_fits_2p5x": bool(2.5 * blk_mb * 1e6 <= budget)}
            try:
                recC["ms"] = round(timed(dev, mkC) * 1e3, 4)
                if baseA:
                    recC["vs_A"] = round(baseA / recC["ms"], 3)
            except Exception as e:                                            # noqa: BLE001
                recC["error"] = str(e)[:200]
            emit(check(mkC, "C", recC))

            # --- D: one block live at a time -- the size-INDEPENDENT form ----------------------
            def mkD(R=R):
                outs = []
                for s in range(0, S, R):
                    b = x[:, s:min(s + R, S), :]
                    p = ttnn.permute(b, (1, 0, 2), memory_config=L1)
                    ttnn.deallocate(b)
                    outs.append(ttnn.to_memory_config(p, DRAM))
                    ttnn.deallocate(p)
                o = ttnn.concat(outs, dim=0, memory_config=DRAM)
                for t_ in outs:
                    ttnn.deallocate(t_)
                return o
            recD = {"N": N, "variant": "D_one_block_l1_spill_concat", "R": R,
                    "calls": 3 * nblk + 1, "blocks": nblk, "block_MB": round(blk_mb, 2),
                    "peak_l1_MB": round(blk_mb, 2)}
            try:
                recD["ms"] = round(timed(dev, mkD) * 1e3, 4)
                if baseA:
                    recD["vs_A"] = round(baseA / recD["ms"], 3)
            except Exception as e:                                            # noqa: BLE001
                recD["error"] = str(e)[:200]
            emit(check(mkD, "D", recD))

            # --- E: blocked but DRAM destination -- control, isolates the L1 destination -------
            def mkE(R=R):
                outs = []
                for s in range(0, S, R):
                    b = x[:, s:min(s + R, S), :]
                    outs.append(ttnn.permute(b, (1, 0, 2), memory_config=DRAM))
                    ttnn.deallocate(b)
                o = ttnn.concat(outs, dim=0, memory_config=DRAM)
                for t_ in outs:
                    ttnn.deallocate(t_)
                return o
            recE = {"N": N, "variant": "E_blocks_dram_concat", "R": R, "calls": 2 * nblk + 1,
                    "blocks": nblk, "peak_l1_MB": 0.0}
            try:
                recE["ms"] = round(timed(dev, mkE) * 1e3, 4)
                if baseA:
                    recE["vs_A"] = round(baseA / recE["ms"], 3)
            except Exception as e:                                            # noqa: BLE001
                recE["error"] = str(e)[:200]
            emit(check(mkE, "E", recE))

        ttnn.deallocate(x)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
